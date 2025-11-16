#!/usr/bin/env python3
"""
prod_dsl_testgen.py
Production-grade DSL v2 → Z3 → PyReason → Testcase generator
with Azure OpenAI extraction + Gradio review UI + unit tests.

Usage:
  - Create .env with AZURE_OPENAI_* keys (endpoint, key, deployment)
  - pip install openai z3-solver pyreason python-dotenv jsonschema gradio pytest
  - python prod_dsl_testgen.py --ui          # runs Gradio UI for review & generate
  - python prod_dsl_testgen.py --run-example # run headless example pipeline
  - pytest prod_dsl_testgen.py               # run unit tests
"""

import os, re, json, textwrap, csv, argparse, logging
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
import openai
from jsonschema import validate as js_validate, ValidationError
from z3 import (
    Solver, Bool, Int, Real, StringVal, String, Length, Re, InRe,
    And, Or, Not, Implies, PrefixOf, SuffixOf, Contains, IntNumRef, BoolRef
)
from pyreason import PyReasoner

# logging config
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("prod_dsl_testgen")

load_dotenv()

# Azure OpenAI config (must be in .env)
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01")

if not (AZURE_ENDPOINT and AZURE_KEY and AZURE_DEPLOYMENT):
    log.warning("Azure OpenAI env vars not fully set; LLM calls will fail unless configured.")

openai.api_type = "azure"
openai.api_base = AZURE_ENDPOINT.rstrip("/") if AZURE_ENDPOINT else None
openai.api_key = AZURE_KEY
openai.api_version = OPENAI_API_VERSION

def azure_chat_completion(prompt: str, temperature=0.0, max_tokens=800) -> str:
    if not openai.api_key or not openai.api_base:
        raise RuntimeError("Azure OpenAI is not configured (check .env variables).")
    resp = openai.ChatCompletion.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message["content"].strip()

# -----------------------------
# DSL v2 JSON Schema (strict)
# -----------------------------
DSL_SCHEMA = {
    "type": "object",
    "properties": {
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "values": {"type": "array"}
                },
                "required": ["name","type"]
            }
        },
        "predicates": {
            "type": "array",
            "items": {
                "type":"object",
                "properties":{
                    "name":{"type":"string"},
                    "args":{"type":"array"},
                    "definition":{"type":"string"}
                },
                "required":["name","args","definition"]
            }
        },
        "constraints": {
            "type": "array",
            "items": {
                "type":"object",
                "properties":{
                    "id":{"type":"string"},
                    "expr":{"type":"string"}
                },
                "required":["id","expr"]
            }
        }
    },
    "required":["variables","constraints"]
}

# -----------------------------
# Helpers: sanitization and splitting
# -----------------------------
def sanitize_name(s: str) -> str:
    return re.sub(r'[^0-9a-zA-Z_]', '_', s)

def split_top_level(s: str, sep: str) -> List[str]:
    depth = 0
    parts = []
    last = 0
    i = 0
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[last:i].strip())
            last = i + len(sep)
            i = last
            continue
        i += 1
    parts.append(s[last:].strip())
    return parts

# -----------------------------
# Deterministic sample value generator
# -----------------------------
def sample_value_for_var(var: Dict[str,Any]) -> Any:
    t = var.get("type","string")
    name = var.get("name","var")
    if t == "bool":
        return False
    if t == "int":
        # pick a typical edge & middle values generator
        return 0 if "age" in name.lower() else 1
    if t == "float":
        return 0.0
    if t == "string":
        if "email" in name.lower():
            return "user@example.com"
        if "password" in name.lower():
            return "Passw0rd"
        return f"{name}_sample"
    if t.startswith("enum"):
        values = var.get("values", [])
        return values[0] if values else (f"{name}_enum")
    if t.startswith("list"):
        # return a JSON-like string representation (we use aux flags for contains)
        return []
    return None

# -----------------------------
# Parse DSL -> build Z3
# -----------------------------
def build_z3_from_dsl(dsl_json: Dict[str,Any]):
    # Validate DSL JSON strictly
    try:
        js_validate(instance=dsl_json, schema=DSL_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(f"DSL JSON failed schema validation: {e.message}")

    variables = dsl_json.get("variables", [])
    predicates = {p['name']: p for p in dsl_json.get("predicates", [])}
    constraints = dsl_json.get("constraints", [])

    z3_vars = {}
    enum_maps = {}
    aux_map = {}  # auxiliary Booolean / Int variable mapping for contains/size/any/all

    # create z3 vars
    for v in variables:
        name = v["name"]
        t = v["type"]
        if t == "bool":
            z3_vars[name] = Bool(name)
        elif t == "int":
            z3_vars[name] = Int(name)
        elif t == "float":
            z3_vars[name] = Real(name)
        elif t == "string":
            z3_vars[name] = String(name)
        elif t.startswith("enum"):
            vals = v.get("values", [])
            enum_maps[name] = {val: idx for idx, val in enumerate(vals)}
            z3_vars[name] = Int(name)
        elif t.startswith("list"):
            # represent list as String placeholder + create size int auxiliary
            z3_vars[name] = String(name)
            size_name = f"{name}__size"
            aux_map[size_name] = Int(size_name)
        elif t.startswith("object"):
            z3_vars[name] = String(name)
        else:
            z3_vars[name] = String(name)

    # expand predicate definitions (careful, expansion will be textual substitution: we only allow arg0.. argN)
    predicate_defs = {}
    for pname, pinfo in predicates.items():
        predicate_defs[pname] = pinfo.get("definition", "")

    # support limited safe grammar: parse atomic and composite expressions
    def parse_atomic(expr: str):
        expr = expr.strip()
        # matches(var, /regex/)
        m = re.match(r'^matches\(\s*([A-Za-z0-9_]+)\s*,\s*/(.+)/\s*\)$', expr)
        if m:
            var = m.group(1)
            regex = m.group(2)
            if var not in z3_vars:
                raise RuntimeError(f"Unknown var in matches(): {var}")
            try:
                return InRe(z3_vars[var], Re(regex))
            except Exception:
                # fallback: create regex as contains '@' check for simple email fallback
                return Contains(z3_vars[var], StringVal("@"))

        # length(var)
        m = re.match(r'^length\(\s*([A-Za-z0-9_]+)\s*\)$', expr)
        if m:
            var = m.group(1)
            if var not in z3_vars:
                raise RuntimeError(f"Unknown var in length(): {var}")
            return Length(z3_vars[var])

        # size(list) -> use aux var if present, else create
        m = re.match(r'^size\(\s*([A-Za-z0-9_]+)\s*\)$', expr)
        if m:
            ln = m.group(1)
            size_name = f"{ln}__size"
            if size_name not in aux_map:
                aux_map[size_name] = Int(size_name)
            return aux_map[size_name]

        # contains(list, 'elem')
        m = re.match(r"^contains\(\s*([A-Za-z0-9_]+)\s*,\s*['\"](.+)['\"]\s*\)$", expr)
        if m:
            var = m.group(1); val = m.group(2)
            aux_name = f"{var}__contains__{sanitize_name(val)}"
            if aux_name not in aux_map:
                aux_map[aux_name] = Bool(aux_name)
            return aux_map[aux_name]

        # starts_with(var, "prefix") -> prefer PrefixOf
        m = re.match(r"^starts_with\(\s*([A-Za-z0-9_]+)\s*,\s*['\"](.+)['\"]\s*\)$", expr)
        if m:
            var = m.group(1); pref = m.group(2)
            if var not in z3_vars:
                raise RuntimeError(f"Unknown var {var} in starts_with")
            try:
                return PrefixOf(StringVal(pref), z3_vars[var])  # PrefixOf(prefix, s)
            except Exception:
                # fallback to Contains at start (weak)
                return Contains(z3_vars[var], StringVal(pref))

        # ends_with(var, "suffix")
        m = re.match(r"^ends_with\(\s*([A-Za-z0-9_]+)\s*,\s*['\"](.+)['\"]\s*\)$", expr)
        if m:
            var = m.group(1); suf = m.group(2)
            try:
                return SuffixOf(StringVal(suf), z3_vars[var])
            except Exception:
                return Contains(z3_vars[var], StringVal(suf))

        # arithmetic or comparison forms: var OP literal OR literal OP var OR literal OP literal
        for op in ["==","!=","<=",">=",">","<"]:
            if op in expr:
                left, right = [x.strip() for x in expr.split(op,1)]
                # handle IN set
                if op == "==" and re.match(r'^[A-Za-z0-9_]+$', left) and re.match(r'^\{.*\}$', right):
                    # left IN {a,b} style with 'IN' will be used instead of ==; fallback
                    pass
                # prepare z3 left and right
                def to_z3(tok):
                    if tok in z3_vars:
                        return z3_vars[tok]
                    if re.match(r'^-?\d+$', tok):
                        return int(tok)
                    if re.match(r'^-?\d+\.\d+$', tok):
                        return float(tok)
                    if tok.lower() in ("true","false"):
                        return True if tok.lower()=="true" else False
                    mstr = re.match(r"^['\"](.+)['\"]$", tok)
                    if mstr:
                        return StringVal(mstr.group(1))
                    # fallback: return tok string (could be enum literal)
                    return tok
                zl = to_z3(left); zr = to_z3(right)
                if op == "==":
                    return zl == zr
                if op == "!=":
                    return zl != zr
                if op == "<=":
                    return zl <= zr
                if op == ">=":
                    return zl >= zr
                if op == "<":
                    return zl < zr
                if op == ">":
                    return zl > zr

        # IN set: var IN {a,b}
        m = re.match(r"^([A-Za-z0-9_]+)\s+IN\s+\{(.+)\}$", expr)
        if m:
            var = m.group(1); vals = [x.strip().strip('\'"') for x in m.group(2).split(",")]
            if var in enum_maps:
                disj = []
                for v in vals:
                    if v not in enum_maps[var]:
                        raise RuntimeError(f"Enum value {v} not in {var}")
                    disj.append(z3_vars[var] == enum_maps[var][v])
                return Or(*disj)
            else:
                disj = []
                for v in vals:
                    disj.append(z3_vars[var] == StringVal(v))
                return Or(*disj)

        # predicate call, e.g., valid_email(email)
        m = re.match(r"^([A-Za-z0-9_]+)\s*\(\s*([A-Za-z0-9_,\s]*)\s*\)$", expr)
        if m:
            pname = m.group(1); args = [a.strip() for a in m.group(2).split(",")] if m.group(2).strip() else []
            if pname in predicate_defs:
                definition = predicate_defs[pname]
                # textual substitution arg0,arg1 -> actual arg names
                for idx, a in enumerate(args):
                    definition = definition.replace(f"arg{idx}", a)
                # parse expanded definition recursively
                return parse_expr(definition)
            else:
                # unsupported predicate call => create a Bool auxiliary var
                aux_name = f"pred_{pname}_{sanitize_name('_'.join(args))}"
                if aux_name not in aux_map:
                    aux_map[aux_name] = Bool(aux_name)
                return aux_map[aux_name]

        # fallback unknown atomic
        # try to parse a bare boolean var
        if expr in z3_vars:
            return z3_vars[expr]
        raise RuntimeError(f"Unsupported atomic expression: {expr}")

    def parse_expr(expr: str):
        expr = expr.strip()
        # IF ... THEN ...
        m = re.match(r'^IF\s+(.+)\s+THEN\s+(.+)$', expr, flags=re.IGNORECASE)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            return Implies(parse_expr(left), parse_expr(right))

        # parentheses
        if expr.startswith("(") and expr.endswith(")"):
            return parse_expr(expr[1:-1])

        # OR
        or_parts = split_top_level(expr, " OR ")
        if len(or_parts) > 1:
            return Or(*[parse_expr(p) for p in or_parts])

        # AND
        and_parts = split_top_level(expr, " AND ")
        if len(and_parts) > 1:
            return And(*[parse_expr(p) for p in and_parts])

        # NOT
        if expr.upper().startswith("NOT "):
            return Not(parse_expr(expr[4:].strip()))

        # any(list, cond) or all(list, cond)
        m = re.match(r'^(any|all)\(\s*([A-Za-z0-9_]+)\s*,\s*(.+)\)$', expr, flags=re.IGNORECASE)
        if m:
            typ = m.group(1).lower(); lst = m.group(2); cond = m.group(3)
            # create auxiliary bool: list_any_<condhash>
            cond_sanit = sanitize_name(cond)
            aux_name = f"{lst}__{typ}__{cond_sanit}"
            if aux_name not in aux_map:
                aux_map[aux_name] = Bool(aux_name)
                # we do NOT expand internal semantics fully; it acts as a selector variable
            return aux_map[aux_name]

        # default to atomic
        return parse_atomic(expr)

    # build constraints
    z3_constraints = []
    for c in constraints:
        cid = c.get("id","")
        expr = c.get("expr","")
        try:
            z3c = parse_expr(expr)
            z3_constraints.append((cid, z3c))
        except Exception as e:
            raise RuntimeError(f"Failed to parse constraint {cid}: {expr}\nReason: {e}")

    # return
    return z3_vars, z3_constraints, aux_map, enum_maps, predicate_defs

# -----------------------------
# Generate models (satisfy + violate)
# -----------------------------
def generate_models(z3_vars, z3_constraints, aux_map, enum_maps, max_pos=5):
    s = Solver()
    for _, c in z3_constraints:
        s.add(c)
    models = []
    # satisfying models
    found = 0
    while found < max_pos and s.check().r == 1:
        m = s.model()
        md = {}
        for k,v in z3_vars.items():
            if v in m:
                val = m[v]
                if isinstance(val, BoolRef):
                    md[k] = bool(val)
                elif isinstance(val, IntNumRef):
                    md[k] = int(val.as_long())
                else:
                    md[k] = str(val)
            else:
                md[k] = None
        # aux map values if present
        for an, av in aux_map.items():
            if av in m:
                val = m[av]
                if isinstance(val, BoolRef):
                    md[an] = bool(val)
                elif isinstance(val, IntNumRef):
                    md[an] = int(val.as_long())
                else:
                    md[an] = str(val)
            else:
                md[an] = None
        models.append({"type":"satisfy_all", "model":md})
        # block exact model
        block_terms = []
        for k,v in z3_vars.items():
            if v in m:
                val = m[v]
                block_terms.append(v != m[v])
        for an,av in aux_map.items():
            if av in m:
                block_terms.append(av != m[av])
        if block_terms:
            s.add(Or(*block_terms))
        found += 1

    # negative models: violate each constraint
    for cid, c in z3_constraints:
        s2 = Solver()
        for iid,cc in z3_constraints:
            if iid != cid:
                s2.add(cc)
        s2.add(Not(c))
        if s2.check().r == 1:
            m = s2.model()
            md = {}
            for k,v in z3_vars.items():
                if v in m:
                    val = m[v]
                    if isinstance(val, BoolRef):
                        md[k] = bool(val)
                    elif isinstance(val, IntNumRef):
                        md[k] = int(val.as_long())
                    else:
                        md[k] = str(val)
                else:
                    md[k] = None
            for an,av in aux_map.items():
                if av in m:
                    val = m[av]
                    if isinstance(val, BoolRef):
                        md[an] = bool(val)
                    elif isinstance(val, IntNumRef):
                        md[an] = int(val.as_long())
                    else:
                        md[an] = str(val)
                else:
                    md[an] = None
            models.append({"type":f"violate_{cid}", "model":md})
    return models

# -----------------------------
# PyReason validation/inference
# -----------------------------
def pyreason_infer_from_rules(pyreason_rules: List[str], solver_model: Dict[str,Any]) -> List[str]:
    engine = PyReasoner()
    for r in pyreason_rules:
        engine.add_rule(r)
    # add facts for solver model
    for k,v in solver_model.items():
        if v is None:
            continue
        if isinstance(v, bool):
            engine.add_fact(f"{k} == {str(v).lower()}.")
        elif isinstance(v, (int, float)):
            engine.add_fact(f"{k} == {v}.")
        else:
            engine.add_fact(f'{k} == "{v}".')
    engine.run()
    return engine.facts()

# -----------------------------
# Convert DSL constraints to PyReason rules (heuristic)
# -----------------------------
def dsl_constraints_to_pyreason(constraints: List[Dict[str,str]]) -> List[str]:
    out = []
    for c in constraints:
        expr = c.get("expr","")
        # IF <a> THEN <b> -> "b :- a."
        m = re.match(r'IF\s+(.+)\s+THEN\s+(.+)', expr, flags=re.IGNORECASE)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            # naive conversion: turn into "right :- left."
            out.append(f"{right} :- {left}.")
    return out

# -----------------------------
# LLM prompts and robust parsing
# -----------------------------
EXTRACT_PROMPT = textwrap.dedent("""
You are a structured extractor. Given a user story + acceptance criteria, output STRICT JSON matching this schema:

{
  "variables":[ {"name":"...","type":"bool|int|float|string|enum|list[...]|object{...}","values":["..."] (for enum) } ],
  "predicates":[ {"name":"pred","args":["type",...],"definition":"..."} ],
  "constraints":[ {"id":"C1","expr":"..."} ]
}

Use only the DSL functions:
 - comparisons: == != > < >= <=
 - boolean ops: AND OR NOT
 - implication: IF <cond> THEN <cond>
 - functions: length(x), matches(x,/regex/), contains(list,'elem'), starts_with(x,'a'), ends_with(x,'z'), size(list), any(list,cond), all(list,cond)
 - enum membership: status IN {active, pending}

Return ONLY JSON (no explanation).
""").strip()

FORMAT_PROMPT = textwrap.dedent("""
You are a test-case generator. Given:
 - user_story (text)
 - constraints (list of DSL expressions in natural form)
 - solver_model (json)
 - inferred_facts (list)
Return a single JSON object:
{ "title":"...", "preconditions":"...", "steps":[ "..."], "inputs": {...}, "expected":"..." }
Return ONLY JSON.
""").strip()

def call_llm_extract(user_story: str) -> Dict[str,Any]:
    prompt = EXTRACT_PROMPT + "\n\nUser story:\n" + user_story
    out = azure_chat_completion(prompt, temperature=0.0, max_tokens=1200)
    # parse JSON strictly
    try:
        parsed = json.loads(out)
    except Exception:
        # try unsafe substring extraction
        m = re.search(r'\{.*\}', out, flags=re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
        else:
            raise RuntimeError("LLM did not return valid JSON for DSL extraction:\n" + out)
    # validate
    js_validate(parsed, DSL_SCHEMA)
    return parsed

def call_llm_format(user_story:str, constraints_nl:List[str], solver_model:Dict[str,Any], inferred_facts:List[str]):
    prompt = FORMAT_PROMPT + "\n\nUser story:\n" + user_story + "\n\nConstraints:\n" + json.dumps(constraints_nl, indent=2)
    prompt += "\n\nSolver model:\n" + json.dumps(solver_model, indent=2) + "\n\nInferred facts:\n" + json.dumps(inferred_facts, indent=2)
    out = azure_chat_completion(prompt, temperature=0.0, max_tokens=600)
    try:
        return json.loads(out)
    except Exception:
        m = re.search(r'\{.*\}', out, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError("LLM did not return valid JSON for testcase formatting:\n" + out)

# -----------------------------
# CSV audit export
# -----------------------------
def export_testcases_csv(testcases: List[Dict[str,Any]], path: str):
    keys = ["title","preconditions","steps","inputs","expected","_meta"]
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for tc in testcases:
            row = [tc.get(k, "") if k != "_meta" else json.dumps(tc.get("_meta",""), ensure_ascii=False) for k in keys]
            writer.writerow(row)
    log.info(f"Exported {len(testcases)} testcases to {path}")

# -----------------------------
# High-level pipeline
# -----------------------------
def pipeline(user_story: str, human_review=False) -> List[Dict[str,Any]]:
    log.info("Calling LLM to extract DSL...")
    dsl = call_llm_extract(user_story)

    # Optionally do human review: here we just trust the DSL if human_review==False
    # Integration with Gradio UI is provided below (if running --ui).

    z3_vars, z3_constraints, aux_map, enum_maps, predicate_defs = build_z3_from_dsl(dsl)
    # include aux map vars into z3 vars for consistent model extraction
    for an,av in aux_map.items():
        # av is already a z3 variable (Bool or Int), but include its name for mapping
        pass

    log.info("Generating models...")
    models = generate_models(z3_vars, z3_constraints, aux_map, enum_maps, max_pos=3)

    # convert constraints to PyReason rules
    pyreason_rules = dsl_constraints_to_pyreason(dsl.get("constraints",[]))
    # also add predicate definitions converted into PyReason textual rules (naive)
    for pname, pdef in predicate_defs.items():
        # pdef may contain arg0,arg1; we create a textual rule head p(arg0) etc.
        # This is a best-effort conversion for inference/explainability; should be reviewed.
        # Example: valid_email(arg0) :- matches(arg0, /.../).
        pass

    testcases = []
    for m in models:
        solver_model = m["model"]
        inferred = pyreason_infer_from_rules(pyreason_rules, solver_model) if pyreason_rules else []
        # populate missing sample values for None variables so LLM formatting receives realistic inputs
        # Build inputs map with sampled values
        inputs_map = {}
        for v in dsl.get("variables", []):
            name = v["name"]
            if name in solver_model and solver_model[name] is not None:
                inputs_map[name] = solver_model[name]
            else:
                inputs_map[name] = sample_value_for_var(v)
        # attach aux flags if any
        for an in aux_map.keys():
            if an in solver_model and solver_model[an] is not None:
                inputs_map[an] = solver_model[an]
        # ask LLM to format final testcase (batching may be advisable in production)
        tc_json = call_llm_format(user_story, [c["expr"] for c in dsl.get("constraints",[])], inputs_map, inferred)
        tc_json["_meta"] = {"solver_model": solver_model, "inferred_facts": inferred, "type": m["type"]}
        testcases.append(tc_json)

    return testcases

# -----------------------------
# Simple Gradio UI for review & generation
# -----------------------------
def launch_ui():
    try:
        import gradio as gr
    except Exception:
        raise RuntimeError("gradio is required for UI. pip install gradio")

    def extract_and_edit(user_story):
        dsl = call_llm_extract(user_story)
        # present DSL JSON for user editing
        return json.dumps(dsl, indent=2, ensure_ascii=False)

    def generate_from_edited(user_story, edited_dsl_json):
        # parse edited JSON (human can modify)
        try:
            dsl = json.loads(edited_dsl_json)
            js_validate(dsl, DSL_SCHEMA)
        except Exception as e:
            return f"DSL JSON invalid: {e}"
        # run generation using given DSL (skip LLM extraction)
        # bypass call_llm_extract -> directly use build_z3_from_dsl
        z3_vars, z3_constraints, aux_map, enum_maps, predicate_defs = build_z3_from_dsl(dsl)
        models = generate_models(z3_vars, z3_constraints, aux_map, enum_maps, max_pos=3)
        pyreason_rules = dsl_constraints_to_pyreason(dsl.get("constraints",[]))
        results = []
        for m in models:
            solver_model = m["model"]
            inferred = pyreason_infer_from_rules(pyreason_rules, solver_model)
            inputs_map = {}
            for v in dsl.get("variables", []):
                name = v["name"]
                inputs_map[name] = solver_model.get(name, None) or sample_value_for_var(v)
            for an in aux_map.keys():
                if an in solver_model:
                    inputs_map[an] = solver_model[an]
            tc_json = call_llm_format(user_story, [c["expr"] for c in dsl.get("constraints",[])], inputs_map, inferred)
            tc_json["_meta"] = {"solver_model": solver_model, "inferred": inferred, "type": m["type"]}
            results.append(tc_json)
        return json.dumps(results, indent=2, ensure_ascii=False)

    with gr.Blocks() as demo:
        gr.Markdown("# DSL v2 Testcase Generator — Human Review UI")
        story = gr.Textbox(lines=8, label="User Story + Acceptance Criteria", value="As a user, ...")
        extract_btn = gr.Button("Extract DSL (Azure OpenAI)")
        dsl_out = gr.Textbox(lines=20, label="Extracted DSL (editable JSON)")
        gen_btn = gr.Button("Generate Testcases from Edited DSL")
        result_out = gr.Textbox(lines=20, label="Generated Testcases JSON")

        extract_btn.click(fn=extract_and_edit, inputs=story, outputs=dsl_out)
        gen_btn.click(fn=generate_from_edited, inputs=[story, dsl_out], outputs=result_out)

        gr.Markdown("**Workflow:** Edit extracted DSL if needed, then press Generate.")
        demo.launch()
