"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
from pathlib import Path

from assistant.infra.config import vocabulary
from assistant.infra.grounding import (
    canonical_name,
    find_codes,
    find_names,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)


def _validate_payload(payload: object) -> dict:
    """Проверяет структуру JSON-ответа модели."""

    if not isinstance(payload, dict):
        raise ValueError("Ответ должен быть JSON-объектом")

    entities = payload.get("entities")
    relations = payload.get("relations")

    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError("entities и relations должны быть списками")

    for item in entities:
        if not isinstance(item, dict):
            raise ValueError("Сущность должна быть объектом")

        if not isinstance(item.get("type"), str) or not isinstance(item.get("name"), str):
            raise ValueError("У сущности должны быть строковые type и name")

        if item["type"] not in ENTITY_TYPES:
            raise ValueError("Неизвестный тип сущности")

    relation_fields = (
        "subject_type",
        "subject",
        "type",
        "object_type",
        "object",
    )

    for item in relations:
        if not isinstance(item, dict):
            raise ValueError("Связь должна быть объектом")

        if not all(isinstance(item.get(field), str) for field in relation_fields):
            raise ValueError("У связи отсутствуют обязательные строковые поля")

        if item["subject_type"] not in ENTITY_TYPES:
            raise ValueError("Неизвестный тип субъекта")

        if item["object_type"] not in ENTITY_TYPES:
            raise ValueError("Неизвестный тип объекта")

        if item["type"] not in RELATION_SIGNATURES:
            raise ValueError("Неизвестный тип отношения")

    return payload


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """

    vocab = vocabulary()

    codes_in_text = set(find_codes(chunk.text))
    names_in_text = set(find_names(chunk.text, vocab))

    prompt_path = Path(__file__).parent / "prompts" / "extraction.txt"

    prompt_template = prompt_path.read_text(encoding="utf-8")

    prompt = prompt_template.replace(
        "{{VOCABULARY}}",
        json.dumps(vocab, ensure_ascii=False, indent=2),
    )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Извлеки сущности и отношения из следующего документа:\n\n{chunk.text}",
        },
    ]

    payload = None

    for attempt in range(2):
        try:
            response = llm.chat(
                messages,
                json_schema=EXTRACTION_SCHEMA,
            )

            payload = _validate_payload(json.loads(response.content))
            break

        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            if attempt == 1:
                return ExtractionResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                )

            messages.extend(
                [
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": (
                            f"Предыдущий ответ некорректен: {error}. "
                            "Исправь его и верни полный JSON строго по заданной схеме."
                        ),
                    },
                ]
            )

    entities_by_id: dict[str, Entity] = {}
    dropped = 0

    for item in payload["entities"]:
        entity_type = item["type"]
        raw_name = item["name"]

        name = canonical_name(entity_type, raw_name, vocab)

        if name is None:
            dropped += 1
            continue

        if entity_type in {"Model", "Part", "ErrorCode", "Procedure"}:
            if name not in codes_in_text:
                dropped += 1
                continue

        else:
            if (entity_type, name) not in names_in_text:
                dropped += 1
                continue

        entity = Entity(entity_type, name)
        entities_by_id[entity.node_id] = entity

    relations = []
    seen_relations = set()

    for item in payload["relations"]:
        subject_type = item["subject_type"]
        object_type = item["object_type"]

        subject_name = canonical_name(
            subject_type,
            item["subject"],
            vocab,
        )
        object_name = canonical_name(
            object_type,
            item["object"],
            vocab,
        )

        if subject_name is None or object_name is None:
            dropped += 1
            continue

        if subject_type in {"Model", "Part", "ErrorCode", "Procedure"}:
            if subject_name not in codes_in_text:
                dropped += 1
                continue
        elif (subject_type, subject_name) not in names_in_text:
            dropped += 1
            continue

        if object_type in {"Model", "Part", "ErrorCode", "Procedure"}:
            if object_name not in codes_in_text:
                dropped += 1
                continue
        elif (object_type, object_name) not in names_in_text:
            dropped += 1
            continue

        subject = Entity(subject_type, subject_name)
        obj = Entity(object_type, object_name)

        relation = Relation(
            subject=subject,
            type=item["type"],
            object=obj,
        )

        if not relation.matches_schema():
            dropped += 1
            continue

        entities_by_id.setdefault(subject.node_id, subject)
        entities_by_id.setdefault(obj.node_id, obj)

        relation_key = (
            subject.node_id,
            relation.type,
            obj.node_id,
        )

        if relation_key not in seen_relations:
            relations.append(relation)
            seen_relations.add(relation_key)

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=list(entities_by_id.values()),
        relations=relations,
        dropped=dropped,
    )
