from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class LiteratureGrounding:
    source: str
    target_paper: str
    reference_titles: list[str]
    design_highlights: list[str]
    dataset_items: list[str]
    metric_items: list[str]
    existing_methods_summary: str
    experiment_plan_summary: str
    weak_context_scaffold: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _first_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _split_outside_parentheses(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _looks_noisy_sentence(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return True
    if re.search(r"\b\d{2,}:\d{1,2}\b", cleaned):
        return True
    non_ascii_count = sum(1 for ch in cleaned if ord(ch) > 127)
    if non_ascii_count > 2:
        return True
    digit_count = sum(1 for ch in cleaned if ch.isdigit())
    if digit_count > 8:
        return True
    noisy_markers = (
        " arxiv:",
        "fig.",
        "figure ",
        "table ",
        "et al .",
        "et al.",
        "doi:",
        "project website",
        "paper introduces",
        "novel dataset captured",
        "captured in diverse real scenarios",
    )
    lowered = cleaned.casefold()
    if any(marker in lowered for marker in noisy_markers):
        return True
    return False


def _join_natural(items: list[str]) -> str:
    cleaned = [item for item in (_clean_text(item) for item in items) if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _clean_topic_prompt(topic: Any) -> str:
    cleaned = _clean_text(topic).rstrip(".")
    for prefix in ("The topic of this paper is ", "Ideation topic keyword: "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned or _clean_text(topic)


def _benchmark_packet(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("benchmark_input_packet", {})
    return payload if isinstance(payload, dict) else {}


def _benchmark(metadata: dict[str, Any]) -> str:
    packet = _benchmark_packet(metadata)
    return _clean_text(metadata.get("benchmark") or packet.get("benchmark"))


def _keyword(metadata: dict[str, Any], topic: Any = "") -> str:
    packet = _benchmark_packet(metadata)
    return _clean_text(metadata.get("keyword") or packet.get("keyword") or _clean_topic_prompt(topic))


def _is_liveideabench_boilerplate(text: str) -> bool:
    lowered = _clean_text(text).casefold()
    return lowered.startswith("benchmark keyword:") or lowered.startswith(
        "this benchmark row provides a keyword prompt"
    ) or lowered.startswith("use the keyword as the ideation seed")


def _reference_titles(literature: list[str], metadata: dict[str, Any]) -> list[str]:
    titles = metadata.get("reference_titles", [])
    extracted: list[str] = []
    if isinstance(titles, list):
        extracted.extend(
            _clean_text(item)
            for item in titles
            if _clean_text(item) and not _is_liveideabench_boilerplate(_clean_text(item))
        )
    for item in literature:
        title = _clean_text(str(item).split("|", 1)[0])
        if title and not _is_liveideabench_boilerplate(title):
            extracted.append(title)
    return _unique_strings(extracted)


def _raw_record(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("raw_record", {})
    return payload if isinstance(payload, dict) else {}


def _summary(metadata: dict[str, Any]) -> dict[str, Any]:
    raw_record = _raw_record(metadata)
    payload = raw_record.get("summary", {})
    return payload if isinstance(payload, dict) else {}


def _method_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(metadata)
    payload = summary.get("method", {})
    return payload if isinstance(payload, dict) else {}


def _target_paper(metadata: dict[str, Any]) -> str:
    return _clean_text(metadata.get("target_paper", ""))


def _paper_grounding(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("paper_grounding", {})
    return payload if isinstance(payload, dict) else {}


def _target_paper_snippet(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = _paper_grounding(metadata).get("target_paper_snippet", {})
    return payload if isinstance(payload, dict) else {}


def _reference_paper_snippets(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _paper_grounding(metadata).get("reference_paper_snippets", [])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _method_summary(metadata: dict[str, Any]) -> str:
    value = _clean_text(metadata.get("method_summary", ""))
    if value:
        return value
    target_snippet = _target_paper_snippet(metadata)
    for key in ("method", "abstract", "introduction", "text_excerpt"):
        value = _clean_text(target_snippet.get(key, ""))
        if value:
            return value
    return _clean_text(_method_payload(metadata).get("targeted_designs_summary", ""))


def _reference_snippet_signal_text(
    metadata: dict[str, Any],
    *,
    preferred_fields: tuple[str, ...],
    limit: int = 4,
) -> str:
    snippets = _reference_paper_snippets(metadata)[:limit]
    fragments: list[str] = []
    for snippet in snippets:
        for field_name in preferred_fields:
            value = _clean_text(snippet.get(field_name, ""))
            if value:
                fragments.append(value)
    return " ".join(fragments)


def _datasets_text(metadata: dict[str, Any]) -> str:
    value = _clean_text(_method_payload(metadata).get("datasets", ""))
    if value:
        return value
    return _reference_snippet_signal_text(
        metadata,
        preferred_fields=("evaluation", "method", "abstract", "introduction", "text_excerpt"),
    )


def _metrics_text(metadata: dict[str, Any]) -> str:
    value = _clean_text(_method_payload(metadata).get("metrics", ""))
    if value:
        return value
    return _reference_snippet_signal_text(
        metadata,
        preferred_fields=("evaluation", "method", "abstract", "introduction", "text_excerpt"),
    )


def _design_highlights(metadata: dict[str, Any], *, limit: int = 3) -> list[str]:
    payload = _method_payload(metadata)
    details = payload.get("targeted_designs_details", [])
    if isinstance(details, list) and details:
        highlights: list[str] = []
        for item in details[:limit]:
            if not isinstance(item, dict):
                continue
            name = _clean_text(item.get("design_name", ""))
            description = _first_sentence(_clean_text(item.get("description", "")))
            if name and description:
                highlights.append(f"{name}: {description}")
            elif name:
                highlights.append(name)
            elif description:
                highlights.append(description)
        return _unique_strings(highlights)[:limit]

    snippet_highlights: list[str] = []
    for snippet in _reference_paper_snippets(metadata)[:limit]:
        resolved_title = _clean_text(
            snippet.get("resolved_title", "")
            or snippet.get("raw_title", "")
            or snippet.get("title", "")
        )
        method = _first_sentence(
            _clean_text(snippet.get("method", ""))
            or _clean_text(snippet.get("abstract", ""))
            or _clean_text(snippet.get("evaluation", ""))
            or _clean_text(snippet.get("introduction", ""))
            or _clean_text(snippet.get("snippet", ""))
        )
        if (
            resolved_title
            and method
            and len(resolved_title) <= 140
            and 40 <= len(method) <= 220
            and not _looks_noisy_sentence(method)
            and "figure" not in method.casefold()
            and "abstract" not in resolved_title.casefold()
        ):
            snippet_highlights.append(f"{resolved_title}: {method}")
    return _unique_strings(snippet_highlights)[:limit]


def _dataset_items(metadata: dict[str, Any]) -> list[str]:
    datasets_text = _datasets_text(metadata)
    if not datasets_text:
        return []

    def _clean_dataset_item(text: str) -> str:
        cleaned = _clean_text(text)
        cleaned = re.sub(r"^evaluate on the\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^evaluate on\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+and report .*$", "", cleaned, flags=re.IGNORECASE)
        if not cleaned or "[" in cleaned or cleaned.casefold().startswith("such methods include"):
            return ""
        lowered = cleaned.casefold()
        if lowered.startswith(("paper introduces ", "this paper introduces ", "we introduce ")):
            return ""
        if "novel dataset captured" in lowered or "captured in diverse real scenarios" in lowered:
            return ""
        if len(cleaned.split()) > 14 and "(" not in cleaned and ")" not in cleaned and "dataset" not in lowered:
            return ""
        if _looks_noisy_sentence(cleaned):
            return ""
        return cleaned.strip(" .")

    explicit_matches = re.findall(
        r"\b([A-Z0-9][A-Za-z0-9\-\+ ]{1,80}?(?:dataset|Dataset|Metropolis|PanoSUNCG|Polycam|ScanNet|KITTI|COCO|ImageNet|Cityscapes|360VO))\b",
        datasets_text,
    )
    explicit_cleaned = _unique_strings(_clean_dataset_item(item) for item in explicit_matches if _clean_dataset_item(item))
    sentence = _first_sentence(datasets_text)
    prefix_candidates = (
        "Experiments were conducted on several datasets, including ",
        "The datasets include ",
        "Datasets include ",
        "Evaluate on the ",
        "Evaluate on ",
    )
    clause = sentence
    for prefix in prefix_candidates:
        if clause.startswith(prefix):
            clause = clause[len(prefix) :]
            break
    clause = clause.rstrip(".")
    clause = clause.replace(", and ", ", ")
    clause = re.sub(r"^and\s+", "", clause, flags=re.IGNORECASE)

    items: list[str] = []
    for item in _split_outside_parentheses(clause):
        cleaned = _clean_dataset_item(item)
        cleaned = re.sub(r"^and\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^custom datasets?\s+", "", cleaned, flags=re.IGNORECASE)
        if re.match(r"^(captured using|which includes|including |are used for evaluation)", cleaned, flags=re.IGNORECASE):
            continue
        if cleaned:
            items.append(cleaned)
    return _unique_strings(explicit_cleaned + items)[:6]


def _metric_items(metadata: dict[str, Any]) -> list[str]:
    metrics_text = _metrics_text(metadata)
    if not metrics_text:
        return []
    matches = re.findall(r"([A-Z][A-Za-z\- ]+?) \(([A-Z]{2,8})\)", metrics_text)
    if matches:
        cleaned_items = []
        for name, abbr in matches:
            cleaned_name = re.sub(r"^Evaluation metrics include\s+", "", name.strip(), flags=re.IGNORECASE)
            candidate = f"{cleaned_name} ({abbr})"
            if not _looks_noisy_sentence(candidate):
                cleaned_items.append(candidate)
        extras = re.findall(
            r"\b(?:mIoU|IoU|PSNR|SSIM|LPIPS|RRE|RTAE|RSE|ARE|ATE|accuracy|precision|recall|F1)\b",
            metrics_text,
            flags=re.IGNORECASE,
        )
        filtered_extras = [
            item
            for item in extras
            if len(_clean_text(item)) > 2 and not _looks_noisy_sentence(item)
        ]
        return _unique_strings(cleaned_items + filtered_extras)

    sentence = _first_sentence(metrics_text)
    if not sentence:
        return []
    prefix_candidates = (
        "Evaluation metrics include ",
        "Metrics include ",
    )
    clause = sentence
    for prefix in prefix_candidates:
        if clause.startswith(prefix):
            clause = clause[len(prefix) :]
            break
    clause = clause.rstrip(".").replace(", and ", ", ")
    items = []
    for item in _split_outside_parentheses(clause):
        cleaned = re.sub(r"\s+for .*$", "", _clean_text(item), flags=re.IGNORECASE)
        if cleaned and len(cleaned) > 2 and not _looks_noisy_sentence(cleaned) and cleaned.casefold() not in {"are", "used"}:
            items.append(cleaned)
    extras = re.findall(
        r"\b(?:mIoU|IoU|PSNR|SSIM|LPIPS|RRE|RTAE|RSE|ARE|ATE|accuracy|precision|recall|F1)\b",
        metrics_text,
        flags=re.IGNORECASE,
    )
    filtered_extras = [
        item
        for item in extras
        if len(_clean_text(item)) > 2 and not _looks_noisy_sentence(item)
    ]
    return _unique_strings(items + filtered_extras)


def _is_keyword_only_context(metadata: dict[str, Any], literature: list[str]) -> bool:
    if _benchmark(metadata).casefold() != "liveideabench":
        return False
    keyword = _keyword(metadata)
    if not keyword:
        return False
    packet = _benchmark_packet(metadata)
    reference_packet = packet.get("reference_packet", []) if isinstance(packet, dict) else []
    if isinstance(reference_packet, list) and reference_packet:
        return False
    if _reference_paper_snippets(metadata):
        return False
    return not _reference_titles(literature, metadata)


def _keyword_domain_family(keyword: str) -> str:
    lowered = keyword.casefold()
    family_terms = {
        "elemental_chemistry": {
            "periodic table", "chemical element", "electron configuration", "atomic radius", "periodic trend",
        },
        "earth_environment": {
            "meteorology", "climate", "ecology", "geology", "oceanography", "hydrology",
            "atmosphere", "greenhouse", "weather", "earthquake", "cartography", "habitat",
        },
        "life_biomed": {
            "neurology", "pathology", "histology", "meiosis", "viruses", "immunotherapy",
            "multiple sclerosis", "endocrinology", "monoclonal antibodies", "biology", "genomics",
            "medicine", "disease", "protein", "cell", "microbiome",
        },
        "chem_materials": {
            "chemistry", "photochemistry", "supramolecular chemistry", "materials", "catalysis",
            "battery", "polymer", "molecule", "spectroscopy", "mass spectrometry",
        },
        "physics_engineering": {
            "mechanics", "optics", "refraction", "microelectronics", "proton", "measurement",
            "petroleum engineering", "supercomputing", "magnetic resonance imaging", "robotics",
        },
        "computation_systems": {
            "linear programming", "geographic information systems", "computer vision", "machine learning",
            "information retrieval", "database", "network", "optimization", "programming",
        },
        "social_human": {
            "cognitive psychology", "diversity in science", "scientific literacy", "economics",
            "education", "policy", "behavior", "humancomputer interaction",
        },
    }
    for family, terms in family_terms.items():
        if any(term in lowered for term in terms):
            return family
    return "general_science"


def _keyword_only_scaffold(metadata: dict[str, Any], topic: Any) -> dict[str, object]:
    keyword = _keyword(metadata, topic)
    if not keyword:
        return {}

    family = _keyword_domain_family(keyword)
    scaffold_by_family: dict[str, dict[str, object]] = {
        "elemental_chemistry": {
            "divergence_axes": [
                "periodic-relation modeling",
                "property prediction across element families",
                "out-of-group generalization across elements",
            ],
            "existing_method_directions": [
                "tabular or graph-based property prediction",
                "periodic-trend representation learning",
                "symbolic or physics-inspired modeling of atomic structure",
            ],
            "design_highlights": [
                "Periodic-Relation Modeling: encode group, period, and valence structure as explicit relations rather than relying only on flat descriptors.",
                "Atomic Attribute Propagation: propagate atomic number, electron configuration, and related descriptors across neighboring elements.",
                "Out-of-Group Generalization: evaluate whether the model extrapolates to held-out element families and rare compounds.",
            ],
            "evaluation_assets": [
                "held-out element property benchmarks",
                "group-wise or period-wise generalization splits",
                "rare-element or compound case studies",
            ],
            "metric_items": ["accuracy", "MAE", "calibration error", "out-of-group accuracy"],
            "risk_items": [
                "memorizing the table instead of learning periodic structure",
                "weak extrapolation to rare elements",
                "mismatch between learned relations and chemical mechanisms",
            ],
            "mechanism_terms": [
                "periodic-relation modeling",
                "atomic attribute propagation",
                "out-of-group generalization",
                "chemical structure priors",
            ],
            "method_instantiation": (
                "Use a relation graph over elements with group and period edges, and run message passing "
                "to propagate atomic attributes such as atomic number and electron configuration."
            ),
        },
        "earth_environment": {
            "divergence_axes": [
                "spatiotemporal forecasting",
                "multi-source sensing and data assimilation",
                "uncertainty-aware extreme-event analysis",
            ],
            "existing_method_directions": [
                "spatiotemporal forecasting models",
                "physics-aware simulation or data-assimilation pipelines",
                "multi-source environmental data fusion",
            ],
            "design_highlights": [
                "Physics-Guided Forecasting: combine learned prediction with domain constraints or dynamical consistency.",
                "Multi-Source Data Fusion: integrate satellite, radar, station, or reanalysis signals to improve coverage.",
                "Uncertainty Calibration: model rare events and regional shift with calibrated confidence estimates.",
            ],
            "evaluation_assets": [
                "reanalysis-based forecasting tasks",
                "satellite and radar nowcasting benchmarks",
                "regional severe-weather case studies",
            ],
            "metric_items": ["RMSE", "MAE", "CRPS", "event F1", "calibration error"],
            "risk_items": [
                "regional distribution shift",
                "rare-event imbalance",
                "tradeoffs between physical consistency and predictive flexibility",
            ],
            "mechanism_terms": [
                "physics-guided forecasting",
                "data fusion",
                "spatiotemporal forecasting",
                "uncertainty calibration",
                "extreme-event modeling",
                "reanalysis",
                "satellite",
                "radar",
            ],
            "method_instantiation": (
                "Use a spatiotemporal encoder to fuse satellite, radar, and reanalysis inputs, "
                "together with a physics-consistency loss that penalizes dynamically implausible forecasts."
            ),
        },
        "life_biomed": {
            "divergence_axes": [
                "mechanism discovery",
                "multimodal representation learning",
                "robust decision support under data scarcity",
            ],
            "existing_method_directions": [
                "biological mechanism modeling",
                "multi-omics or multimodal fusion",
                "clinical or experimental decision support",
            ],
            "design_highlights": [
                "Mechanism-Aware Modeling: encode pathway or structural priors rather than using only black-box predictors.",
                "Multimodal Fusion: integrate heterogeneous measurements such as imaging, sequences, or assays.",
                "Robust Decision Support: emphasize uncertainty, subgroup robustness, and low-sample generalization.",
            ],
            "evaluation_assets": [
                "held-out cohort or assay benchmarks",
                "cross-lab or cross-population transfer tasks",
                "mechanism-grounded case studies",
            ],
            "metric_items": ["AUROC", "AUPRC", "F1", "calibration error", "subgroup robustness"],
            "risk_items": [
                "small-sample overfitting",
                "distribution shift across cohorts",
                "weak biological interpretability",
            ],
            "mechanism_terms": [
                "mechanism-aware",
                "multimodal fusion",
                "cohort robustness",
                "pathway prior",
                "uncertainty-aware",
            ],
        },
        "chem_materials": {
            "divergence_axes": [
                "property prediction",
                "structure-conditioned generation or search",
                "mechanism-aware simulation acceleration",
            ],
            "existing_method_directions": [
                "property-prediction models",
                "structure-aware inverse design",
                "simulation-accelerated screening",
            ],
            "design_highlights": [
                "Structure-Conditioned Search: couple candidate generation with explicit structural validity checks.",
                "Property-Aware Representation: encode molecular or material structure for target-property prediction.",
                "Simulation-Guided Filtering: use coarse simulation or surrogate checks before expensive evaluation.",
            ],
            "evaluation_assets": [
                "property-prediction benchmarks",
                "structure-validity screening tasks",
                "candidate-ranking case studies",
            ],
            "metric_items": ["MAE", "RMSE", "top-k hit rate", "validity", "novelty"],
            "risk_items": [
                "distribution shift to novel chemistries",
                "validity-performance tradeoffs",
                "surrogate mismatch with real simulation",
            ],
            "mechanism_terms": [
                "structure-conditioned",
                "property prediction",
                "simulation-guided",
                "candidate ranking",
            ],
        },
        "physics_engineering": {
            "divergence_axes": [
                "inverse modeling",
                "constraint-aware control or design",
                "measurement-efficient inference",
            ],
            "existing_method_directions": [
                "physics-constrained inference",
                "control or optimization under constraints",
                "measurement-efficient estimation",
            ],
            "design_highlights": [
                "Constraint-Aware Inference: enforce conservation laws, geometry, or hardware limits in the model.",
                "Measurement-Efficient Estimation: trade off sensing cost against prediction quality.",
                "Robust Control or Design: stress test under perturbations, noise, or operating shifts.",
            ],
            "evaluation_assets": [
                "simulation-to-real transfer tasks",
                "noise-robust estimation benchmarks",
                "resource-constrained design case studies",
            ],
            "metric_items": ["RMSE", "constraint violation", "latency", "energy cost", "robustness under noise"],
            "risk_items": [
                "sim-to-real mismatch",
                "constraint violation under shift",
                "resource-quality tradeoffs",
            ],
            "mechanism_terms": [
                "constraint-aware",
                "measurement-efficient",
                "robust control",
                "inverse modeling",
            ],
        },
        "computation_systems": {
            "divergence_axes": [
                "resource-efficient optimization",
                "structured representation or retrieval",
                "robust decision-making under shift",
            ],
            "existing_method_directions": [
                "optimization or search algorithms",
                "representation learning and retrieval",
                "robust or adaptive decision systems",
            ],
            "design_highlights": [
                "Structured Search: decompose the problem into explicit subproblems instead of one monolithic predictor.",
                "Resource-Aware Adaptation: trade off accuracy against latency, memory, or compute.",
                "Robust Generalization: evaluate under distribution shift, noisy inputs, or adversarial settings.",
            ],
            "evaluation_assets": [
                "held-out benchmark suites",
                "out-of-distribution or shift benchmarks",
                "latency-constrained deployment tasks",
            ],
            "metric_items": ["accuracy", "F1", "latency", "memory use", "robustness under shift"],
            "risk_items": [
                "overfitting to benchmark heuristics",
                "resource-performance tradeoffs",
                "weak generalization out of distribution",
            ],
            "mechanism_terms": [
                "structured search",
                "resource-aware adaptation",
                "robust generalization",
                "retrieval",
                "optimization",
            ],
        },
        "social_human": {
            "divergence_axes": [
                "measurement and assessment",
                "intervention design",
                "heterogeneity and fairness analysis",
            ],
            "existing_method_directions": [
                "measurement and predictive assessment",
                "behavioral or policy intervention studies",
                "fairness and subgroup analysis",
            ],
            "design_highlights": [
                "Measurement-Aware Modeling: define explicit constructs before optimizing predictors.",
                "Intervention-Oriented Design: connect model outputs to actionable interventions.",
                "Heterogeneity Analysis: quantify subgroup differences, fairness, or contextual variation.",
            ],
            "evaluation_assets": [
                "cross-context benchmark studies",
                "subgroup or fairness analyses",
                "intervention simulation tasks",
            ],
            "metric_items": ["accuracy", "F1", "calibration error", "fairness gap", "policy utility"],
            "risk_items": [
                "construct mismatch",
                "subgroup harms",
                "weak transfer across populations",
            ],
            "mechanism_terms": [
                "measurement-aware",
                "intervention-oriented",
                "heterogeneity analysis",
                "fairness",
            ],
        },
        "general_science": {
            "divergence_axes": [
                "prediction",
                "mechanism understanding",
                "robust evaluation under shift",
            ],
            "existing_method_directions": [
                "predictive modeling",
                "mechanism-aware representation learning",
                "robust or uncertainty-aware evaluation",
            ],
            "design_highlights": [
                "Mechanism-Aware Modeling: add an explicit inductive bias tied to the keyword rather than a generic predictor.",
                "Multiview Integration: combine complementary data views or evidence sources when available.",
                "Robust Evaluation: test under shift, uncertainty, and component ablations instead of only a headline score.",
            ],
            "evaluation_assets": [
                "held-out benchmark tasks",
                "out-of-distribution or stress-test splits",
                "domain-grounded case studies",
            ],
            "metric_items": ["accuracy", "F1", "calibration error", "robustness under shift"],
            "risk_items": [
                "overly generic task formulations",
                "distribution shift",
                "weak linkage between the mechanism and the domain structure",
            ],
            "mechanism_terms": [
                "mechanism-aware",
                "multiview integration",
                "robust evaluation",
                "uncertainty-aware",
            ],
        },
    }
    selected = dict(scaffold_by_family.get(family, scaffold_by_family["general_science"]))
    selected["keyword"] = keyword
    selected["domain_family"] = family
    selected["topic"] = _clean_topic_prompt(topic)
    return selected


def build_literature_grounding(
    *,
    literature: list[str],
    metadata: dict[str, Any],
) -> LiteratureGrounding:
    reference_titles = _reference_titles(literature, metadata)[:6]
    target_paper = _target_paper(metadata)
    method_summary = _method_summary(metadata)
    design_highlights = _design_highlights(metadata)
    dataset_items = _dataset_items(metadata)
    metric_items = _metric_items(metadata)
    weak_context_scaffold = (
        _keyword_only_scaffold(metadata, metadata.get("topic", ""))
        if _is_keyword_only_context(metadata, literature)
        else {}
    )
    if weak_context_scaffold:
        if not design_highlights:
            design_highlights = _unique_strings(
                [_clean_text(item) for item in weak_context_scaffold.get("design_highlights", [])]
            )[:3]
        if not dataset_items:
            dataset_items = _unique_strings(
                [_clean_text(item) for item in weak_context_scaffold.get("evaluation_assets", [])]
            )[:4]
        if not metric_items:
            metric_items = _unique_strings(
                [_clean_text(item) for item in weak_context_scaffold.get("metric_items", [])]
            )[:6]
    paper_grounding = _paper_grounding(metadata)
    paper_grounding_source = "paper_snippets" if paper_grounding.get("reference_paper_snippets") or paper_grounding.get("target_paper_snippet") else ""

    existing_parts: list[str] = []
    if reference_titles:
        existing_parts.append(
            "The provided literature context includes "
            + _join_natural(reference_titles[:4])
            + "."
        )
    if method_summary:
        method_sentence = _first_sentence(method_summary)
        if target_paper and method_sentence.casefold().startswith(target_paper.casefold()):
            existing_parts.append(method_sentence)
        elif target_paper:
            existing_parts.append(
                f"The benchmark target paper {target_paper} can be summarized as follows: {method_sentence}"
            )
        else:
            existing_parts.append(f"A representative target method can be summarized as follows: {method_sentence}")
    if design_highlights:
        existing_parts.append(
            "Notable design elements in this context include "
            + _join_natural([item.rstrip(".") for item in design_highlights[:3]])
            + "."
        )
    if weak_context_scaffold:
        keyword = _clean_text(weak_context_scaffold.get("keyword"))
        directions = _unique_strings(
            [_clean_text(item) for item in weak_context_scaffold.get("existing_method_directions", [])]
        )[:3]
        divergence_axes = _unique_strings(
            [_clean_text(item) for item in weak_context_scaffold.get("divergence_axes", [])]
        )[:3]
        if directions:
            existing_parts.append(
                f"For {keyword}, safe starting directions include " + _join_natural(directions) + "."
            )
        if divergence_axes:
            existing_parts.append(
                "For divergent thinking under weak context, prioritize branches around "
                + _join_natural(divergence_axes)
                + "."
            )
    if not existing_parts:
        existing_parts.append(
            "Only limited literature context is available, so the existing-method summary is provisional."
        )

    experiment_parts: list[str] = []
    if dataset_items:
        experiment_parts.append("Evaluate on " + _join_natural(dataset_items) + ".")
    if metric_items:
        experiment_parts.append("Report " + _join_natural(metric_items) + ".")
    if weak_context_scaffold:
        risk_items = _unique_strings(
            [_clean_text(item) for item in weak_context_scaffold.get("risk_items", [])]
        )[:2]
        if risk_items:
            experiment_parts.append("Stress test " + _join_natural(risk_items) + ".")
    if not experiment_parts:
        experiment_parts.append(
            "Compare against strong baselines using task-relevant datasets, ablations, and quantitative metrics."
        )

    if paper_grounding_source:
        source = paper_grounding_source
    elif weak_context_scaffold:
        source = "keyword_scaffold"
    elif any([method_summary, dataset_items, metric_items, design_highlights]):
        source = "metadata_structured"
    else:
        source = "titles_only"
    return LiteratureGrounding(
        source=source,
        target_paper=target_paper,
        reference_titles=reference_titles,
        design_highlights=design_highlights,
        dataset_items=dataset_items,
        metric_items=metric_items,
        existing_methods_summary=" ".join(existing_parts),
        experiment_plan_summary=" ".join(experiment_parts),
        weak_context_scaffold=weak_context_scaffold,
    )
