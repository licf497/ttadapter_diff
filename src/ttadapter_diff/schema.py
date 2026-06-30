from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import joblib

@dataclass
class TritonKernelMetadata:
    kernel_name: str
    module_name: str
    kernel_path: Path
    rel_kernel_path: Path
    signature: dict[str, str]
    constants: dict[str, Any]
    target: Any
    options: Any
    call_stack: list[str]
    sys_argv: list[str] = field(default_factory=list)

    @property
    def content(self) -> dict[str, any]:
        return {
            "kernel_name": self.kernel_name,
            "rel_kernel_path": self.rel_kernel_path.as_posix() if isinstance(self.rel_kernel_path, Path) else self.rel_kernel_path,
            "signature": self.signature,
            "constants": self.constants,
            "target": self.target,
            "options": self.options,
        }

    @property
    def content_hash(self) -> str | None:
        return joblib.hash(self.content, hash_name='md5')

@dataclass
class CompileResult:
    ttadapter_str: str
    meta: TritonKernelMetadata