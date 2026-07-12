"""Case-directory construction: source staging + template substitution +
per-param input_fns.

This unifies the (previously duplicated) directory-preparation machinery of
the legacy ``GridManager.prepare_case`` and ``MCMCManager._dispatch_walker``.

Each case directory is built by running these steps *in order*:

1. **Stage source dir** — if ``cfg.case_source_dir`` is set, its contents are
   recursively copied into the case directory first, and every copied file
   whose path matches ``cfg.case_substitute_globs`` has ``@TOKEN@``
   substitution applied in place (see :meth:`CaseBuilder._stage_source_dir`).
2. **Render templates** — ``@KEY@`` tokens in every template file are
   substituted with the parameter values (upper-cased names, ``%.10g``
   formatting) plus ``@JOB_NAME@``. A trailing ``.template`` suffix is
   stripped from the destination file name. ``*.sh`` templates (or templates
   already executable) are made executable. Templates are written into the
   case directory *after* staging, so a template can overwrite a
   same-named file that was staged from the source directory.
3. **input_fns** — each param's ``input_fn(case_dir, value,
   **selected_params)`` is invoked with keyword arguments filtered by
   signature introspection.
4. **file_pipeline** — if ``cfg.file_pipeline`` is non-empty,
   ``utils.file_pipeline.apply_pipeline`` runs last, generating/modifying
   extra files declared in the YAML config.
"""

from __future__ import annotations

import fnmatch
import inspect
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from varify.src.common.config import FrameworkConfig
from varify.utils.file_pipeline import apply_pipeline


class CaseBuilder:
    """Prepares a single simulation case directory from templates."""

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self._log = logging.getLogger("varify.casebuilder")

    @staticmethod
    def substitute(text: str, smap: Dict[str, str]) -> str:
        for key, val in smap.items():
            text = text.replace(f"@{key}@", val)
        return text

    @staticmethod
    def substitution_map(params: Dict[str, float], job_name: str) -> Dict[str, str]:
        m: Dict[str, str] = {"JOB_NAME": job_name}
        for name, val in params.items():
            m[name.upper()] = f"{val:.10g}"
        return m

    def _stage_source_dir(self, case_dir: Path, smap: Dict[str, str]) -> None:
        """Copy ``cfg.case_source_dir`` into *case_dir*, then apply @TOKEN@
        substitution in place to every copied file matching
        ``cfg.case_substitute_globs``.

        Matching is against both the bare filename and the file's path
        relative to *case_dir* (POSIX separators), so globs like
        ``"*.inp"`` and ``"sub/*.inp"`` both work. Binary files (that fail
        UTF-8 decoding) are left untouched. Executable bits are preserved by
        ``shutil.copytree``.
        """
        src = self.cfg.case_source_dir
        if src is None:
            return
        if not src.is_dir():
            self._log.warning(
                "case_source_dir configured but not found, skipping: %s", src
            )
            return
        shutil.copytree(src, case_dir, dirs_exist_ok=True)

        globs = self.cfg.case_substitute_globs
        for path in case_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(case_dir).as_posix()
            if not any(
                fnmatch.fnmatch(path.name, g) or fnmatch.fnmatch(rel, g)
                for g in globs
            ):
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            filled = self.substitute(raw, smap)
            if filled != raw:
                path.write_text(filled, encoding="utf-8")

    def _render_templates(self, case_dir: Path, smap: Dict[str, str]) -> None:
        for tpl_path_str in self.cfg.template_files:
            tpl_path = Path(tpl_path_str)
            if not tpl_path.exists():
                self._log.warning("Template not found, skipping: %s", tpl_path)
                continue
            raw = tpl_path.read_text(encoding="utf-8")
            filled = self.substitute(raw, smap)
            dest_name = tpl_path.name
            if dest_name.endswith(".template"):
                dest_name = dest_name[: -len(".template")]
            dest = case_dir / dest_name
            dest.write_text(filled, encoding="utf-8")
            if tpl_path.suffix == ".sh" or os.access(tpl_path, os.X_OK):
                dest.chmod(dest.stat().st_mode | 0o111)

    def _run_input_fns(self, case_dir: Path, params: Dict[str, float]) -> None:
        for spec in self.cfg.param_specs:
            if spec.input_fn is None:
                continue
            val = params[spec.name]
            try:
                sig = inspect.signature(spec.input_fn)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                declared_kw = [
                    n for n, p in sig.parameters.items()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ) and n not in ("case_dir", "value")
                ]
                extra: Dict[str, Any] = params.copy() if has_var_kw else {
                    k: params[k] for k in declared_kw if k in params
                }
                spec.input_fn(case_dir, val, **extra)
            except Exception as exc:
                self._log.error(
                    "input_fn(%s) FAILED for %s: %s", spec.name, case_dir.name, exc
                )

    def build(
        self,
        case_dir: Path,
        params: Dict[str, float],
        job_name: str,
    ) -> Path:
        """Create *case_dir* and run the full per-case pipeline, in order:
        stage source dir -> render templates -> input_fns -> file_pipeline.
        """
        case_dir.mkdir(parents=True, exist_ok=True)
        smap = self.substitution_map(params, job_name)
        self._stage_source_dir(case_dir, smap)
        self._render_templates(case_dir, smap)
        self._run_input_fns(case_dir, params)
        if self.cfg.file_pipeline:
            apply_pipeline(case_dir, params, self.cfg.file_pipeline)
        self._log.debug("Prepared: %s", case_dir)
        return case_dir
