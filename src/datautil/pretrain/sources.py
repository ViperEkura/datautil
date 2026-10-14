"""Compose document record mappings with shared file storage backends."""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from datautil.common.io import iter_raw, output_writer
from datautil.common.storage import storage_for
from datautil.pretrain.formats import DocumentFormatAdapter
from datautil.pretrain.formats import validate_fields as validate_fields
from datautil.pretrain.model import DocumentRecord


def iter_documents(
    path: str, source: str, fields: Dict[str, Any], fmt: Optional[str] = None
) -> Iterator[DocumentRecord]:
    adapter = DocumentFormatAdapter(fields, source)
    for location, raw in iter_raw(path, storage_for(path, fmt or "auto")):
        yield adapter.parse(raw, location)


@contextmanager
def document_writer(path: str, storage: str = "auto") -> Iterator[Any]:
    adapter = DocumentFormatAdapter()
    with output_writer(path, storage) as writer:
        yield lambda record: writer.write(adapter.serialize(record))
