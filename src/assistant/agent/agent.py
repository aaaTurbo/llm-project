"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from pathlib import Path

from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSED_TEXT = "В базе знаний не найдено."
UNAVAILABLE_TEXT = "Данные ERP получить не удалось."

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")
CORRECTION_PROMPT = (PROMPTS_DIR / "correction.txt").read_text(encoding="utf-8")

CODE_PATTERN = re.compile(
    r"\b(?:P-\d{4}[A-Z]?|(?:AR|BZ|GS)-\d{3}|E-\d{2,3}|SP-\d{3})\b"
)
PRICE_PATTERN = re.compile(
    r"(\d[\d ]*)\s*(?:руб(?:\.|лей|ля)?|₽)",
    re.IGNORECASE,
)
STOCK_PATTERN = re.compile(
    r"(\d+)\s*(?:шт(?:\.|ук[аи]?)?|ед(?:\.|иниц[аы]?)?)",
    re.IGNORECASE,
)
PART_PATTERN = re.compile(r"\bP-\d{4}[A-Z]?\b", re.IGNORECASE)
MODEL_PATTERN = re.compile(r"\b(?:AR|BZ|GS)-\d{3}\b", re.IGNORECASE)
ORDER_PATTERN = re.compile(
    r"\b(?:"
    r"закажи(?:те)?|"
    r"оформи(?:те)?\s+заказ|"
    r"оформить\s+заказ|"
    r"создай(?:те)?\s+заказ|"
    r"создать\s+заказ|"
    r"размести(?:те)?\s+заказ|"
    r"разместить\s+заказ|"
    r"хочу\s+заказать"
    r")\b",
    re.IGNORECASE,
)
GLOBAL_PATTERN = re.compile(
    r"(?:чаще\s+всего|больше\s+всего|наибольш(?:ий|ая|ее|его)|"
    r"сняты\s+с\s+производства|сводк|сравн(?:и|ить|ение))",
    re.IGNORECASE,
)
REPLACEMENT_PATTERN = re.compile(
    r"(?P<source>P-\d{4}[A-Z]?) -REPLACED_BY-> (?P<target>P-\d{4}[A-Z]?)"
)
HAS_PART_PATTERN = re.compile(
    r"(?P<model>(?:AR|BZ|GS)-\d{3}) -HAS_PART-> (?P<part>P-\d{4}[A-Z]?)"
)
PART_TYPE_PATTERN = re.compile(
    r"(?P<part>P-\d{4}[A-Z]?) -IS_A-> (?P<component>.+)"
)
ERROR_COMPONENT_PATTERN = re.compile(
    r"(?P<error>E-\d{2,3}) -INDICATES-> (?P<component>.+)"
)
MODEL_COMPONENT_PATTERN = re.compile(
    r"(?P<model>(?:AR|BZ|GS)-\d{3}) -FAILED-> (?P<component>.+)"
)


def _codes(text: str) -> set[str]:
    return {match.group(0).upper() for match in CODE_PATTERN.finditer(text)}


def _collect_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_collect_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_collect_text(item) for item in value)
    return ""


def _number_from_match(value: str) -> int:
    return int(value.replace(" ", ""))


def _question_numbers(question: str) -> set[int]:
    cleaned = CODE_PATTERN.sub(" ", question)
    return {
        int(value)
        for value in re.findall(r"(?<!\w)\d+(?!\w)", cleaned)
    }


def _has_quantity(question: str, qty: int) -> bool:
    return qty in _question_numbers(question)


def _is_global_question(question: str) -> bool:
    return bool(GLOBAL_PATTERN.search(question))


def _is_direct_erp_question(question: str) -> bool:
    has_part = bool(PART_PATTERN.search(question))
    asks_price = bool(re.search(r"цен|стоимост|сколько стоит", question, re.IGNORECASE))
    asks_stock = bool(re.search(r"налич|остат|на складе|сколько штук", question, re.IGNORECASE))
    return has_part and (asks_price or asks_stock)


def _valid_final_answer(
    answer: str,
    question: str,
    observations: list[dict],
) -> bool:
    """Проверяет коды, цены и остатки в ответе по вопросу и результатам инструментов."""
    known_text = question + " " + _collect_text(observations)
    known_codes = _codes(known_text)

    if not _codes(answer).issubset(known_codes):
        return False

    known_prices = {
        int(item["price_rub"])
        for item in observations
        if isinstance(item, dict)
        and isinstance(item.get("price_rub"), (int, float))
        and not isinstance(item.get("price_rub"), bool)
    }
    for match in PRICE_PATTERN.finditer(answer):
        if _number_from_match(match.group(1)) not in known_prices:
            return False

    known_stock = {
        int(item["qty"])
        for item in observations
        if isinstance(item, dict)
        and isinstance(item.get("qty"), int)
        and not isinstance(item.get("qty"), bool)
    }
    allowed_stock = known_stock | _question_numbers(question)

    for match in STOCK_PATTERN.finditer(answer):
        if int(match.group(1)) not in allowed_stock:
            return False

    return True


def _unique_sources(observations: list[dict]) -> list[str]:
    sources = set()
    for item in observations:
        if isinstance(item, dict):
            values = item.get("sources", [])
            if isinstance(values, list):
                sources.update(
                    source for source in values if isinstance(source, str)
                )
    return sorted(sources)


def _tool_error(call: ToolCall, message: str) -> dict:
    return {"error": message}


def _run_tool(
    tool_name: str,
    args: dict,
    tools: dict[str, Callable[..., dict]],
) -> dict:
    result = None
    error = None

    for attempt in range(3):
        try:
            result = tools[tool_name](**args)
            break
        except ErpUnavailable as exc:
            error = str(exc)
            if attempt == 2:
                result = {"error": UNAVAILABLE_TEXT}
        except Exception as exc:
            error = str(exc)
            result = {"error": error}
            break

    if result is None:
        result = {"error": error or "Инструмент не вернул результат."}
    if not isinstance(result, dict):
        result = {"error": "Инструмент вернул результат неверного формата."}
    return result


def _knowledge_facts(observations: list[dict]) -> list[str]:
    facts = []
    for item in observations:
        if isinstance(item, dict) and isinstance(item.get("facts"), list):
            facts.extend(value for value in item["facts"] if isinstance(value, str))
    return facts


def _latest_part(part_id: str, replacements: dict[str, str]) -> str:
    current = part_id
    visited = {current}
    while current in replacements and replacements[current] not in visited:
        current = replacements[current]
        visited.add(current)
    return current


def _resolve_order_part(question: str, observations: list[dict]) -> str | None:
    explicit = PART_PATTERN.findall(question)
    replacements = {}
    facts = _knowledge_facts(observations)
    model_matches = MODEL_PATTERN.findall(question)
    error_matches = re.findall(r"\bE-\d{2,3}\b", question, re.IGNORECASE)
    model_parts = []
    part_components = {}
    error_components = set()
    model_components = set()

    for fact in facts:
        replacement = REPLACEMENT_PATTERN.search(fact)
        if replacement:
            replacements[replacement.group("source").upper()] = replacement.group("target").upper()

        has_part = HAS_PART_PATTERN.search(fact)
        if has_part and (not model_matches or has_part.group("model").upper() in {item.upper() for item in model_matches}):
            model_parts.append(has_part.group("part").upper())

        part_type = PART_TYPE_PATTERN.search(fact)
        if part_type:
            part_components[part_type.group("part").upper()] = part_type.group("component").strip().casefold()

        error_component = ERROR_COMPONENT_PATTERN.search(fact)
        if error_component and error_component.group("error").upper() in {item.upper() for item in error_matches}:
            error_components.add(error_component.group("component").strip().casefold())

        model_component = MODEL_COMPONENT_PATTERN.search(fact)
        if model_component and (not model_matches or model_component.group("model").upper() in {item.upper() for item in model_matches}):
            model_components.add(model_component.group("component").strip().casefold())

    if explicit:
        return _latest_part(explicit[0].upper(), replacements)

    question_lower = question.casefold()
    requested_components = {
        component
        for component in part_components.values()
        if component and component in question_lower
    }
    matching_components = error_components.intersection(model_components)
    if not matching_components:
        matching_components = error_components or model_components or requested_components
    elif requested_components:
        matching_components = matching_components.intersection(requested_components) or matching_components

    matching_parts = [
        part_id for part_id in model_parts
        if part_components.get(part_id) in matching_components
    ]
    candidates = matching_parts or model_parts

    if not candidates:
        return None

    unique_candidates = sorted(set(candidates))
    return _latest_part(unique_candidates[0], replacements)


def _order_arguments(question: str, observations: list[dict]) -> dict | None:
    qty_values = sorted(_question_numbers(question))
    model_match = MODEL_PATTERN.search(question)
    warehouse = next((name for name in WAREHOUSES if name.casefold() in question.casefold()), None)
    part_id = _resolve_order_part(question, observations)

    if not qty_values or not model_match or not warehouse or not part_id:
        return None

    return {
        "part_id": part_id,
        "warehouse": warehouse,
        "qty": qty_values[0],
        "model": model_match.group(0).upper(),
    }


def _result_step(tool_name: str, args: dict, result: dict) -> ToolStep:
    result_error = result.get("error")
    if result_error:
        return ToolStep(
            tool=tool_name,
            args=args,
            result=result,
            error=str(result_error),
        )
    return ToolStep(tool=tool_name, args=args, result=result)


def _not_found_result(result: dict) -> bool:
    error = str(result.get("error", "")).casefold()
    return "деталь не найдена в erp" in error or "не найден в erp" in error


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    if max_steps < 1:
        raise ValueError("max_steps должен быть не меньше 1.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps: list[ToolStep] = []
    observations: list[dict] = []
    executed: dict[str, dict] = {}
    known_codes = _codes(question)
    tool_calls_count = 0

    while tool_calls_count < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS, max_tokens=1024)
        messages.append(response.as_message())

        if not response.tool_calls:
            if not steps:
                if _is_direct_erp_question(question):
                    requested_part = PART_PATTERN.search(question)
                    if requested_part:
                        part_id = requested_part.group(0).upper()
                        asks_price = bool(re.search(r"цен|стоимост|сколько стоит", question, re.IGNORECASE))
                        asks_stock = bool(re.search(r"налич|остат|на складе|сколько штук", question, re.IGNORECASE))
                        warehouse = next((name for name in WAREHOUSES if name.casefold() in question.casefold()), None)
                        direct_calls = []
                        if asks_price and "get_price" in tools:
                            direct_calls.append(("get_price", {"part_id": part_id}))
                        if asks_stock and warehouse and "check_stock" in tools:
                            direct_calls.append(("check_stock", {"part_id": part_id, "warehouse": warehouse}))
                        for direct_tool, direct_args in direct_calls:
                            tool_calls_count += 1
                            direct_result = _run_tool(direct_tool, direct_args, tools)
                            observations.append(direct_result)
                            steps.append(_result_step(direct_tool, direct_args, direct_result))
                            if _not_found_result(direct_result):
                                return AgentResult(
                                    answer=f"Деталь {part_id} не найдена в ERP.",
                                    steps=steps,
                                    sources=_unique_sources(observations),
                                    refused=True,
                                )
                        if direct_calls:
                            direct_results = [item for item in observations if item]
                            price_result = next((item for item in direct_results if "price_rub" in item), None)
                            stock_result = next((item for item in direct_results if "qty" in item and item.get("warehouse") == warehouse), None)
                            parts = []
                            if asks_price:
                                if price_result is not None:
                                    parts.append(f"Цена детали {part_id}: {price_result['price_rub']} руб.")
                                else:
                                    parts.append(UNAVAILABLE_TEXT)
                            if asks_stock and warehouse:
                                if stock_result is not None:
                                    parts.append(f"Остаток на складе {warehouse}: {stock_result['qty']} шт.")
                                else:
                                    parts.append(UNAVAILABLE_TEXT)
                            direct_answer = " ".join(parts) or UNAVAILABLE_TEXT
                            return AgentResult(
                                answer=direct_answer,
                                steps=steps,
                                sources=_unique_sources(observations),
                                refused=False,
                            )
                else:
                    lookup_tool = "summarize_topic" if _is_global_question(question) else "search_knowledge"
                    if lookup_tool in tools:
                        lookup_args = {"question": question}
                        lookup_result = _run_tool(lookup_tool, lookup_args, tools)
                        observations.append(lookup_result)
                        steps.append(_result_step(lookup_tool, lookup_args, lookup_result))
                        tool_calls_count += 1
                        if lookup_result.get("refused") is True and not lookup_result.get("facts"):
                            return AgentResult(
                                answer=REFUSED_TEXT,
                                steps=steps,
                                sources=_unique_sources(observations),
                                refused=True,
                            )
                        asks_price = bool(re.search(r"цен|стоимост|сколько стоит", question, re.IGNORECASE))
                        asks_stock = bool(re.search(r"налич|остат|на складе|сколько штук", question, re.IGNORECASE))
                        if asks_price or asks_stock:
                            resolved_part = _resolve_order_part(question, observations)
                            warehouse = next((name for name in WAREHOUSES if name.casefold() in question.casefold()), None)
                            erp_calls = []
                            if resolved_part and asks_price and "get_price" in tools:
                                erp_calls.append(("get_price", {"part_id": resolved_part}))
                            if resolved_part and asks_stock and warehouse and "check_stock" in tools:
                                erp_calls.append(("check_stock", {"part_id": resolved_part, "warehouse": warehouse}))
                            for erp_tool, erp_args in erp_calls:
                                tool_calls_count += 1
                                erp_result = _run_tool(erp_tool, erp_args, tools)
                                observations.append(erp_result)
                                steps.append(_result_step(erp_tool, erp_args, erp_result))
                                if _not_found_result(erp_result):
                                    return AgentResult(
                                        answer=f"Деталь {resolved_part} не найдена в ERP.",
                                        steps=steps,
                                        sources=_unique_sources(observations),
                                        refused=True,
                                    )
                            if erp_calls:
                                erp_results = observations[1:]
                                price_result = next((item for item in erp_results if "price_rub" in item), None)
                                stock_result = next((item for item in erp_results if "qty" in item and item.get("warehouse") == warehouse), None)
                                answer_parts = [f"Рекомендуемая деталь: {resolved_part}."]
                                if asks_price:
                                    if price_result is not None:
                                        answer_parts.append(f"Цена: {price_result['price_rub']} руб.")
                                    else:
                                        answer_parts.append(UNAVAILABLE_TEXT)
                                if asks_stock and warehouse:
                                    if stock_result is not None:
                                        answer_parts.append(f"Остаток на складе {warehouse}: {stock_result['qty']} шт.")
                                    else:
                                        answer_parts.append(UNAVAILABLE_TEXT)
                                erp_answer = " ".join(answer_parts)
                                return AgentResult(
                                    answer=erp_answer,
                                    steps=steps,
                                    sources=_unique_sources(observations),
                                    refused=False,
                                )
                        if ORDER_PATTERN.search(question):
                            if not _question_numbers(question):
                                return AgentResult(
                                    answer="Сколько штук нужно заказать?",
                                    steps=steps,
                                    sources=_unique_sources(observations),
                                    refused=False,
                                )
                            order_args = _order_arguments(question, observations)
                            if order_args is not None and not confirm:
                                return AgentResult(
                                    answer=(
                                        f"Подтвердите заказ {order_args['qty']} шт. "
                                        f"детали {order_args['part_id']} на склад "
                                        f"{order_args['warehouse']} для модели {order_args['model']}."
                                    ),
                                    steps=steps,
                                    sources=_unique_sources(observations),
                                    refused=False,
                                    needs_confirmation=True,
                                    pending_action={"tool": "create_service_order", "args": order_args},
                                )
                        return AgentResult(
                            answer=str(lookup_result.get("answer", REFUSED_TEXT)),
                            steps=steps,
                            sources=_unique_sources(observations),
                            refused=bool(lookup_result.get("refused", False)),
                        )

            if ORDER_PATTERN.search(question):
                if not _question_numbers(question):
                    return AgentResult(
                        answer="Сколько штук нужно заказать?",
                        steps=steps,
                        sources=_unique_sources(observations),
                        refused=False,
                    )

                order_args = _order_arguments(question, observations)
                if order_args is not None:
                    if not confirm:
                        return AgentResult(
                            answer=(
                                f"Подтвердите заказ {order_args['qty']} шт. "
                                f"детали {order_args['part_id']} на склад "
                                f"{order_args['warehouse']} для модели {order_args['model']}."
                            ),
                            steps=steps,
                            sources=_unique_sources(observations),
                            refused=False,
                            needs_confirmation=True,
                            pending_action={"tool": "create_service_order", "args": order_args},
                        )

                    if "create_service_order" in tools:
                        cache_key = json.dumps(
                            {"tool": "create_service_order", "args": order_args},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        result = executed.get(cache_key)
                        if result is None:
                            result = _run_tool("create_service_order", order_args, tools)
                            executed[cache_key] = result
                        steps.append(_result_step("create_service_order", order_args, result))
                        observations.append(result)
                        if not result.get("error"):
                            order_id = result.get("order_id")
                            answer = f"Заказ {order_id} создан." if order_id else "Заказ создан."
                            return AgentResult(
                                answer=answer,
                                steps=steps,
                                sources=_unique_sources(observations),
                                refused=False,
                            )

            answer = response.content.strip() or "Не удалось сформировать ответ."
            if "в базе знаний не найдено" in answer.casefold() and not any(
                step.tool in ("get_price", "check_stock") for step in steps
            ):
                knowledge_answers = [
                    item.get("answer")
                    for item in observations
                    if isinstance(item, dict)
                    and item.get("facts")
                    and not item.get("refused")
                    and isinstance(item.get("answer"), str)
                    and item.get("answer").strip()
                ]
                if knowledge_answers:
                    answer = knowledge_answers[-1]

            if not _valid_final_answer(answer, question, observations):
                messages.append({"role": "system", "content": CORRECTION_PROMPT})
                correction = llm.chat(messages, max_tokens=1024)
                corrected = correction.content.strip()

                if not corrected or not _valid_final_answer(corrected, question, observations):
                    return AgentResult(
                        answer=REFUSED_TEXT,
                        steps=steps,
                        sources=_unique_sources(observations),
                        refused=True,
                    )
                answer = corrected

            return AgentResult(
                answer=answer,
                steps=steps,
                sources=_unique_sources(observations),
                refused=False,
            )

        for call in response.tool_calls:
            if tool_calls_count >= max_steps:
                break

            tool_calls_count += 1
            args = call.arguments if isinstance(call.arguments, dict) else {"_raw": call.arguments}
            tool_name = call.name

            if "_raw" in args:
                message = (
                    "Не удалось разобрать аргументы инструмента. "
                    "Передай корректные аргументы в формате JSON."
                )
                steps.append(ToolStep(tool=tool_name, args=args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            actual_tool_name = tool_name
            actual_args = args
            if tool_name == "search_knowledge" and _is_global_question(question):
                actual_tool_name = "summarize_topic"
                actual_args = {"question": question}

            if actual_tool_name not in tools:
                message = f"Неизвестный инструмент: {actual_tool_name}."
                steps.append(ToolStep(tool=actual_tool_name, args=actual_args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            if actual_tool_name == "create_service_order" and not ORDER_PATTERN.search(question):
                message = (
                    "Пользователь не просил оформлять заказ. "
                    "Не вызывай create_service_order. Продолжи отвечать "
                    "на исходный вопрос."
                )
                steps.append(ToolStep(tool=actual_tool_name, args=actual_args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            part_id = actual_args.get("part_id")
            if part_id is not None and str(part_id).upper() not in known_codes:
                message = (
                    f"Номер детали {part_id} не встречался в вопросе или "
                    "результатах инструментов. Сначала вызови search_knowledge, "
                    "чтобы найти подтверждённый номер детали."
                )
                steps.append(ToolStep(tool=actual_tool_name, args=actual_args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            if actual_tool_name == "create_service_order":
                qty = actual_args.get("qty")
                if isinstance(qty, bool) or not isinstance(qty, int) or not _has_quantity(question, qty):
                    message = (
                        "Пользователь не указал это количество. "
                        "Уточни, сколько деталей нужно заказать."
                    )
                    steps.append(ToolStep(tool=actual_tool_name, args=actual_args, error=message))
                    return AgentResult(
                        answer="Сколько штук нужно заказать?",
                        steps=steps,
                        sources=_unique_sources(observations),
                        refused=False,
                    )

                resolved_args = _order_arguments(question, observations)
                if resolved_args is not None:
                    actual_args = resolved_args

                if not confirm:
                    return AgentResult(
                        answer=(
                            f"Подтвердите заказ {actual_args['qty']} шт. "
                            f"детали {actual_args['part_id']} на склад "
                            f"{actual_args['warehouse']} для модели {actual_args['model']}."
                        ),
                        steps=steps,
                        sources=_unique_sources(observations),
                        refused=False,
                        needs_confirmation=True,
                        pending_action={"tool": actual_tool_name, "args": actual_args},
                    )

            cache_key = json.dumps(
                {"tool": actual_tool_name, "args": actual_args},
                ensure_ascii=False,
                sort_keys=True,
            )

            if cache_key in executed:
                result = executed[cache_key]
            else:
                result = _run_tool(actual_tool_name, actual_args, tools)
                executed[cache_key] = result

            observations.append(result)
            known_codes.update(_codes(_collect_text(result)))
            step = _result_step(actual_tool_name, actual_args, result)
            steps.append(step)

            if _not_found_result(result):
                missing_part = actual_args.get("part_id", "Деталь")
                return AgentResult(
                    answer=f"Деталь {missing_part} не найдена в ERP.",
                    steps=steps,
                    sources=_unique_sources(observations),
                    refused=True,
                )

            if (
                actual_tool_name in ("search_knowledge", "summarize_topic")
                and result.get("refused") is True
                and not result.get("facts")
            ):
                return AgentResult(
                    answer=REFUSED_TEXT,
                    steps=steps,
                    sources=_unique_sources(observations),
                    refused=True,
                )

            messages.append(tool_message(call, result))

            if actual_tool_name == "create_service_order" and not result.get("error"):
                order_id = result.get("order_id")
                answer = f"Заказ {order_id} создан." if order_id else "Заказ создан."
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    sources=_unique_sources(observations),
                    refused=False,
                )


    return AgentResult(
        answer=(
            "Достигнут лимит вызовов инструментов. Работа остановлена; "
            "использованы только полученные данные."
        ),
        steps=steps,
        sources=_unique_sources(observations),
        refused=False,
    )
