import autogen
import json
import z3
import ast
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import hashlib


class InputType(Enum):
    """Types of inputs supported"""
    SOURCE_CODE = "source_code"
    USER_STORY = "user_story"
    BUG_REPORT = "bug_report"
    DOCUMENTATION = "documentation"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    API_SPEC = "api_spec"


@dataclass
class TestInput:
    """Container for various input types"""
    input_type: InputType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=highest, 5=lowest
    
    def get_hash(self) -> str:
        """Generate unique hash for this input"""
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


@dataclass
class TestCase:
    """Enhanced test case with traceability"""
    test_id: str
    test_name: str
    description: str
    input_values: Dict[str, Any]
    expected_output: Any
    test_type: str
    preconditions: List[str]
    postconditions: List[str]
    constraints_satisfied: List[str]
    coverage_score: float
    quality_score: float
    
    # Traceability
    source_inputs: List[str] = field(default_factory=list)  # Input hashes
    user_story_id: Optional[str] = None
    bug_id: Optional[str] = None
    requirement_id: Optional[str] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    priority: int = 1
    mutation_variants: List[Dict] = field(default_factory=list)
    symbolic_proof: str = ""
    review_comments: List[str] = field(default_factory=list)
    generation_iteration: int = 1
    
    def to_dict(self):
        return asdict(self)


class MultiModalTestGenerator:
    """
    Advanced multi-modal neuro-symbolic test generator
    Accepts: source code, user stories, bug reports, documentation, or any combination
    """
    
    def __init__(self, config_path: str = "OAI_CONFIG_LIST"):
        self.config_list = autogen.config_list_from_json(config_path)
        self.test_inputs: List[TestInput] = []
        self.test_cases: List[TestCase] = []
        self.synthesized_knowledge: Dict[str, Any] = {}
        self.traceability_matrix: Dict[str, List[str]] = {}
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized agents for multi-modal input processing"""
        
        # 1. User Story Analyzer Agent
        self.user_story_analyzer = autogen.AssistantAgent(
            name="UserStoryAnalyzer",
            system_message="""You are an expert in analyzing user stories and acceptance criteria.

Your role is to:
1. Parse user stories in formats: As a [user], I want [feature], so that [benefit]
2. Extract acceptance criteria (Given-When-Then format preferred)
3. Identify edge cases and constraints from the story
4. Determine priority and complexity
5. Extract functional and non-functional requirements
6. Identify user personas and their expected behaviors

Output as JSON with:
- story_id: unique identifier
- role: user role/persona
- feature: desired functionality
- benefit: business value
- acceptance_criteria: list of AC in Given-When-Then format
- edge_cases: identified edge scenarios
- constraints: technical/business constraints
- test_scenarios: high-level test scenarios
- priority: HIGH|MEDIUM|LOW
- complexity: SIMPLE|MEDIUM|COMPLEX
""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.2,
                "seed": 42
            }
        )
        
        # 2. Bug Report Analyzer Agent
        self.bug_analyzer = autogen.AssistantAgent(
            name="BugAnalyzer",
            system_message="""You are an expert in analyzing bug reports and defects.

Your role is to:
1. Parse bug reports (title, description, steps to reproduce, expected vs actual)
2. Identify root cause and failure conditions
3. Extract reproduction steps as test scenarios
4. Determine bug severity and priority
5. Identify similar/related bug patterns
6. Suggest regression test requirements

Output as JSON with:
- bug_id: unique identifier
- title: bug summary
- severity: CRITICAL|HIGH|MEDIUM|LOW
- root_cause: identified cause
- reproduction_steps: list of steps
- expected_behavior: what should happen
- actual_behavior: what actually happens
- failure_conditions: conditions triggering the bug
- affected_components: impacted code areas
- regression_test_requirements: tests needed to prevent recurrence
- related_bugs: similar issues
""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.2,
                "seed": 42
            }
        )
        
        # 3. Documentation Parser Agent
        self.doc_parser = autogen.AssistantAgent(
            name="DocumentationParser",
            system_message="""You are an expert in parsing technical documentation.

Your role is to:
1. Extract functional requirements from documentation
2. Identify API contracts, schemas, and interfaces
3. Parse technical specifications and constraints
4. Extract business rules and validation logic
5. Identify integration points and dependencies
6. Extract examples and use cases

Output as JSON with:
- doc_type: API|DESIGN|REQUIREMENTS|TECHNICAL|USER_MANUAL
- requirements: list of extracted requirements
- api_contracts: API specifications if present
- business_rules: validation and business logic rules
- constraints: technical limitations and constraints
- examples: usage examples and scenarios
- dependencies: external systems and integrations
""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.2,
                "seed": 42
            }
        )
        
        # 4. Code Analyzer Agent (enhanced from previous version)
        self.code_analyzer = autogen.AssistantAgent(
            name="CodeAnalyzer",
            system_message="""You are an expert in source code analysis and program understanding.

Your role is to:
1. Analyze function/class signatures and contracts
2. Extract preconditions, postconditions, and invariants
3. Identify input/output specifications
4. Detect error handling and exception paths
5. Analyze control flow and data flow
6. Identify security-critical operations
7. Extract validation logic and constraints

Output as JSON with:
- function_name: main function/class name
- signature: complete signature
- parameters: list with types and constraints
- return_type: return value specification
- preconditions: required conditions before execution
- postconditions: guaranteed conditions after execution
- invariants: maintained properties
- error_conditions: exceptions and error paths
- complexity: computational complexity
- side_effects: state changes, I/O, etc.
""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.1,
                "seed": 42
            }
        )
        
        # 5. Knowledge Synthesizer Agent (NEW)
        self.knowledge_synthesizer = autogen.AssistantAgent(
            name="KnowledgeSynthesizer",
            system_message="""You are an expert in synthesizing information from multiple sources.

Your role is to:
1. Combine insights from code, user stories, bugs, and documentation
2. Resolve conflicts and ambiguities between sources
3. Create a unified understanding of system behavior
4. Identify gaps in specification coverage
5. Prioritize requirements based on all inputs
6. Build comprehensive test requirements

When analyzing multiple inputs:
- Code provides implementation truth
- User stories provide intended behavior
- Bugs reveal failure modes
- Documentation provides specifications

Output as JSON with:
- unified_requirements: consolidated requirements list
- test_priorities: prioritized test scenarios
- conflicts: identified conflicts between sources
- gaps: missing information or specifications
- risk_areas: high-risk functionality needing thorough testing
- traceability: mapping of requirements to sources
""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.3,
                "seed": 42
            }
        )
        
        # 6. Constraint Analyzer (from previous version)
        self.constraint_analyzer = autogen.AssistantAgent(
            name="ConstraintAnalyzer",
            system_message="""You are a symbolic reasoning expert. Extract ALL logical constraints
from provided inputs (code, requirements, bugs, etc.) and build formal constraint models.

Output as structured JSON with preconditions, postconditions, invariants, boundaries,
equivalence_classes, and error_conditions.""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.1,
                "seed": 42
            }
        )
        
        # 7-12. Keep other agents from previous version
        # (Test Generator, Coverage Validator, Quality Reviewer, Refinement Agent, Mutation Generator)
        self._initialize_generation_agents()
        
        # Coordinator
        self.coordinator = autogen.UserProxyAgent(
            name="TestCoordinator",
            system_message="Coordinates multi-modal test generation workflow",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=20,
            code_execution_config={
                "work_dir": "test_workspace",
                "use_docker": False
            }
        )
    
    def _initialize_generation_agents(self):
        """Initialize test generation and quality agents (from previous version)"""
        
        self.test_generator = autogen.AssistantAgent(
            name="TestGenerator",
            system_message="""You are an expert test case generator. Generate comprehensive
test cases covering all scenarios including boundary, edge, negative, and mutation tests.
Output as JSON array of test cases.""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.7,
                "seed": 42
            }
        )
        
        self.coverage_validator = autogen.AssistantAgent(
            name="CoverageValidator",
            system_message="""Analyze test coverage completeness. Calculate coverage scores
and identify missing scenarios. Output as JSON with coverage_scores, missing_scenarios,
and recommendations.""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.2,
                "seed": 42
            }
        )
        
        self.quality_reviewer = autogen.AssistantAgent(
            name="QualityReviewer",
            system_message="""Review test quality on: Correctness, Completeness, Clarity,
Maintainability, Independence, Realism (0-10 each). Provide overall_score,
dimension_scores, issues, suggestions, and approval_status.""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.3,
                "seed": 42
            }
        )
        
        self.refinement_agent = autogen.AssistantAgent(
            name="RefinementAgent",
            system_message="""Refine test cases based on feedback. Fix issues, generate
additional tests for gaps, improve quality. Output refined_tests, new_tests,
and changes_summary as JSON.""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.6,
                "seed": 42
            }
        )
        
        self.mutation_generator = autogen.AssistantAgent(
            name="MutationGenerator",
            system_message="""Generate mutation tests. Create realistic code mutations and
test variants to ensure test effectiveness. Output mutation variants as JSON.""",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.5,
                "seed": 42
            }
        )
    
    # =========================================================================
    # Input Processing Methods
    # =========================================================================
    
    def add_input(self, content: str, input_type: InputType, 
                  metadata: Optional[Dict] = None, priority: int = 1):
        """Add an input of any type"""
        test_input = TestInput(
            input_type=input_type,
            content=content,
            metadata=metadata or {},
            priority=priority
        )
        self.test_inputs.append(test_input)
        print(f"✓ Added {input_type.value} input (priority: {priority})")
    
    def add_source_code(self, code: str, function_name: str = None, priority: int = 1):
        """Add source code to analyze"""
        metadata = {"function_name": function_name} if function_name else {}
        self.add_input(code, InputType.SOURCE_CODE, metadata, priority)
    
    def add_user_story(self, story: str, story_id: str = None, priority: int = 1):
        """Add user story"""
        metadata = {"story_id": story_id} if story_id else {}
        self.add_input(story, InputType.USER_STORY, metadata, priority)
    
    def add_bug_report(self, bug: str, bug_id: str = None, severity: str = "MEDIUM", priority: int = 1):
        """Add bug report"""
        metadata = {"bug_id": bug_id, "severity": severity} if bug_id else {"severity": severity}
        self.add_input(bug, InputType.BUG_REPORT, metadata, priority)
    
    def add_documentation(self, doc: str, doc_type: str = None, priority: int = 2):
        """Add documentation"""
        metadata = {"doc_type": doc_type} if doc_type else {}
        self.add_input(doc, InputType.DOCUMENTATION, metadata, priority)
    
    def add_acceptance_criteria(self, criteria: str, story_id: str = None, priority: int = 1):
        """Add acceptance criteria"""
        metadata = {"story_id": story_id} if story_id else {}
        self.add_input(criteria, InputType.ACCEPTANCE_CRITERIA, metadata, priority)
    
    # =========================================================================
    # Input Analysis Methods
    # =========================================================================
    
    def analyze_user_story(self, story_input: TestInput) -> Dict[str, Any]:
        """Analyze user story using specialized agent"""
        print(f"  → Analyzing user story...")
        
        message = f"""Analyze this user story and extract all testable information:

{story_input.content}

Provide comprehensive analysis including acceptance criteria, edge cases, and test scenarios."""
        
        self.coordinator.initiate_chat(
            self.user_story_analyzer,
            message=message,
            max_turns=2
        )
        
        response = self.coordinator.chat_messages[self.user_story_analyzer][-1]["content"]
        analysis = self._parse_json_response(response)
        analysis["input_hash"] = story_input.get_hash()
        return analysis
    
    def analyze_bug_report(self, bug_input: TestInput) -> Dict[str, Any]:
        """Analyze bug report using specialized agent"""
        print(f"  → Analyzing bug report...")
        
        message = f"""Analyze this bug report and extract test requirements:

{bug_input.content}

Identify reproduction steps, failure conditions, and regression test requirements."""
        
        self.coordinator.initiate_chat(
            self.bug_analyzer,
            message=message,
            max_turns=2
        )
        
        response = self.coordinator.chat_messages[self.bug_analyzer][-1]["content"]
        analysis = self._parse_json_response(response)
        analysis["input_hash"] = bug_input.get_hash()
        return analysis
    
    def analyze_documentation(self, doc_input: TestInput) -> Dict[str, Any]:
        """Analyze documentation using specialized agent"""
        print(f"  → Analyzing documentation...")
        
        message = f"""Parse this documentation and extract testable requirements:

{doc_input.content}

Extract functional requirements, API contracts, business rules, and constraints."""
        
        self.coordinator.initiate_chat(
            self.doc_parser,
            message=message,
            max_turns=2
        )
        
        response = self.coordinator.chat_messages[self.doc_parser][-1]["content"]
        analysis = self._parse_json_response(response)
        analysis["input_hash"] = doc_input.get_hash()
        return analysis
    
    def analyze_source_code(self, code_input: TestInput) -> Dict[str, Any]:
        """Analyze source code using specialized agent"""
        print(f"  → Analyzing source code...")
        
        message = f"""Analyze this source code and extract testable specifications:

