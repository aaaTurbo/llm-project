"""Инструменты агента. Требование FR-6."""


import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

PART_PATTERN = re.compile(r"^P-\d{4}[A-Z]?$")
MODEL_PATTERN = re.compile(r"^(?:AR|BZ|GS)-\d{3}$")


def _error(message: str) -> dict:
    return {"error": message}


def _valid_part(part_id: object) -> bool:
    return isinstance(part_id, str) and bool(PART_PATTERN.fullmatch(part_id))


def _valid_model(model: object) -> bool:
    return isinstance(model, str) and bool(MODEL_PATTERN.fullmatch(model))

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Найти конкретные факты в графе знаний: ошибки, детали, замены, регламенты и связи с моделями.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Вопрос о конкретных фактах или сущностях."}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Получить общий обзор темы или сводку по нескольким сообществам графа знаний.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Вопрос для обзорного поиска."}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить наличие детали на конкретном складе. Нулевой остаток не означает, что детали нет в ERP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-\\d{4}[A-Z]?$"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получить цену детали в рублях и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-\\d{4}[A-Z]?$"},
                },
                "required": ["part_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ детали. Вызывай только если пользователь назвал количество и подтвердил создание заказа.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-\\d{4}[A-Z]?$"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string", "pattern": "^(AR|BZ|GS)-\\d{3}$"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
                "additionalProperties": False,
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def search_knowledge(*, question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return _error("Нужно передать непустой вопрос.")
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(*, question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return _error("Нужно передать непустой вопрос.")
        answer = global_search(question, kb, llm)
        return {
            "answer": answer.text,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def check_stock(*, part_id: str, warehouse: str) -> dict:
        if not _valid_part(part_id):
            return _error("Некорректный номер детали. Ожидается формат P-1042 или P-1042B.")
        if warehouse not in WAREHOUSES:
            return _error(f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}.")

        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound as exc:
            return _error(f"Деталь не найдена в ERP: {exc}")
        except ErpRejected as exc:
            return _error(f"ERP отклонила запрос: {exc}")

    def get_price(*, part_id: str) -> dict:
        if not _valid_part(part_id):
            return _error("Некорректный номер детали. Ожидается формат P-1042 или P-1042B.")

        try:
            return erp.get_price(part_id)
        except ErpNotFound as exc:
            return _error(f"Деталь не найдена в ERP: {exc}")
        except ErpRejected as exc:
            return _error(f"ERP отклонила запрос: {exc}")

    def create_service_order(
        *,
        part_id: str,
        warehouse: str,
        qty: int,
        model: str,
    ) -> dict:
        if not _valid_part(part_id):
            return _error("Некорректный номер детали. Ожидается формат P-1042 или P-1042B.")
        if warehouse not in WAREHOUSES:
            return _error(f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}.")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return _error("Количество должно быть целым числом больше нуля.")
        if not _valid_model(model):
            return _error("Некорректный код модели. Ожидается формат AR-210, BZ-110 или GS-410.")

        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as exc:
            return _error(f"Деталь не найдена в ERP: {exc}")
        except ErpRejected as exc:
            return _error(f"ERP отклонила заказ: {exc}")

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
