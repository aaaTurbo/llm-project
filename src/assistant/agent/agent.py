"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from pathlib import Path

from assistant.agent.tools import TOOL_SCHEMAS
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
ORDER_PATTERN = re.compile(
    r"\b(?:"
    r"закажи(?:те)?|"
    r"оформи(?:те)?\s+заказ|"
    r"оформить\s+заказ|"
    r"создай(?:те)?\s+заказ|"
    r"создать\s+заказ|"
    r"размести(?:те)?\s+заказ|"
    r"разместить\s+заказ|"
    r"хочу\s+заказать|"
    r"нужно\s+заказать"
    r")\b",
    re.IGNORECASE,
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


def _has_quantity(question: str, qty: int) -> bool:
    return any(
        int(value) == qty
        for value in re.findall(r"(?<!\w)\d+(?!\w)", question)
    )


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
    question_numbers = {
        int(value)
        for value in re.findall(r"(?<!\w)\d+(?!\w)", question)
    }
    allowed_stock = known_stock | question_numbers

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
    knowledge_refused = False
    tool_calls_count = 0

    while tool_calls_count < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS, max_tokens=1024)
        messages.append(response.as_message())

        if not response.tool_calls:
            answer = response.content.strip() or "Не удалось сформировать ответ."

            if (
                knowledge_refused
                and answer.rstrip(".!").casefold()
                == REFUSED_TEXT.rstrip(".").casefold()
            ):
                return AgentResult(
                    answer=REFUSED_TEXT,
                    steps=steps,
                    sources=_unique_sources(observations),
                    refused=True,
                )

            if not _valid_final_answer(answer, question, observations):
                messages.append(
                    {"role": "system", "content": CORRECTION_PROMPT}
                )
                correction = llm.chat(messages, max_tokens=1024)
                corrected = correction.content.strip()

                if (
                    not corrected
                    or not _valid_final_answer(
                        corrected, question, observations
                    )
                ):
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
            args = (
                call.arguments
                if isinstance(call.arguments, dict)
                else {"_raw": call.arguments}
            )
            tool_name = call.name

            if "_raw" in args:
                message = (
                    "Не удалось разобрать аргументы инструмента. "
                    "Передай корректные аргументы в формате JSON."
                )
                steps.append(ToolStep(tool=tool_name, args=args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            if tool_name not in tools:
                message = f"Неизвестный инструмент: {tool_name}."
                steps.append(ToolStep(tool=tool_name, args=args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            if tool_name == "create_service_order" and not ORDER_PATTERN.search(question):
                message = (
                    "Пользователь не просил оформлять заказ. "
                    "Не вызывай create_service_order. Продолжи отвечать "
                    "на исходный вопрос."
                )
                steps.append(ToolStep(tool=tool_name, args=args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            part_id = args.get("part_id")
            if part_id is not None and part_id not in known_codes:
                message = (
                    f"Номер детали {part_id} не встречался в вопросе или "
                    "результатах инструментов. Сначала вызови search_knowledge, "
                    "чтобы найти подтверждённый номер детали."
                )
                steps.append(ToolStep(tool=tool_name, args=args, error=message))
                messages.append(tool_message(call, _tool_error(call, message)))
                continue

            if tool_name == "create_service_order":
                qty = args.get("qty")
                if (
                    isinstance(qty, bool)
                    or not isinstance(qty, int)
                    or not _has_quantity(question, qty)
                ):
                    message = (
                        "Пользователь не указал это количество. "
                        "Уточни, сколько деталей нужно заказать."
                    )
                    steps.append(
                        ToolStep(tool=tool_name, args=args, error=message)
                    )
                    return AgentResult(
                        answer="Сколько штук нужно заказать?",
                        steps=steps,
                        sources=_unique_sources(observations),
                        refused=False,
                    )

                if not confirm:
                    return AgentResult(
                        answer=(
                            "Для создания заказа требуется подтверждение "
                            "пользователя."
                        ),
                        steps=steps,
                        sources=_unique_sources(observations),
                        refused=False,
                        needs_confirmation=True,
                        pending_action={"tool": tool_name, "args": args},
                    )

            cache_key = json.dumps(
                {"tool": tool_name, "args": args},
                ensure_ascii=False,
                sort_keys=True,
            )

            if cache_key in executed:
                result = executed[cache_key]
            else:
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
                    result = {
                        "error": error or "Инструмент не вернул результат."
                    }

                executed[cache_key] = result

            if not isinstance(result, dict):
                result = {
                    "error": "Инструмент вернул результат неверного формата."
                }

            observations.append(result)
            known_codes.update(_codes(_collect_text(result)))

            result_error = result.get("error")
            if result_error:
                step = ToolStep(
                    tool=tool_name,
                    args=args,
                    result=result,
                    error=str(result_error),
                )
            else:
                step = ToolStep(
                    tool=tool_name,
                    args=args,
                    result=result,
                )

            steps.append(step)

            if (
                tool_name in ("search_knowledge", "summarize_topic")
                and result.get("refused") is True
                and not result.get("facts")
            ):
                knowledge_refused = True

            messages.append(tool_message(call, result))

            if tool_name == "create_service_order" and not result_error:
                order_id = result.get("order_id")
                answer = (
                    f"Заказ {order_id} создан."
                    if order_id
                    else "Заказ создан."
                )
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
