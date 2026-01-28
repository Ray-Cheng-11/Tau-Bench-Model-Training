"""
Task Configuration & Groundtruth Generation Pipeline

This module implements the multi-stage pipeline described in the project: context
preparation, LLM-based generation, format & execution checking, multi-LLM review
committee, and feedback-driven refinement. The implementation uses existing
components where possible (TauBenchDataReader, TauBenchOpenAIGenerator, TaskValidator)
and provides a simple loop that attempts to refine tasks until they validate.

Enhanced features:
- User ID consistency validation across all actions
- Detailed statistics tracking and reporting
- Progress tracking with timestamps
- Robust error recovery mechanisms
- Blueprint-based generation integration
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import time
import json
import os
from dataclasses import dataclass, field
from collections import defaultdict

from data_reader import TauBenchDataReader
from task_generator import TauBenchOpenAIGenerator, TaskValidator
from configs import TauBenchConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class PipelineConfig:
    envs_path: str = "envs/retail"
    max_iterations: int = 3
    committee_size: int = 3
    min_accepts: int = 2
    include_metadata: bool = True
    save_path: str = "generated_tasks/Sampled_Tasks.json"
    enable_user_id_validation: bool = True
    max_generation_retries: int = 2
    enforce_scenario_diversity: bool = True
    scenario_rotation: List[str] = field(default_factory=lambda: TauBenchConfig().scenario_keys)
    # AgentFlow generation options
    use_agentflow: bool = False
    agentflow_max_turns: int = 5


@dataclass
class PipelineStatistics:
    """Track detailed statistics about the generation pipeline"""
    total_attempts: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    validation_failures: int = 0
    committee_rejections: int = 0
    user_id_inconsistencies: int = 0
    auto_corrections_applied: int = 0
    generation_errors: int = 0
    iterations_per_task: List[int] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def add_failure_reason(self, reason: str):
        """Record a failure reason"""
        self.failure_reasons[reason] += 1
    
    def finalize(self):
        """Mark the pipeline as complete"""
        self.end_time = time.time()
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_tasks / self.total_attempts) * 100
    
    def get_avg_iterations(self) -> float:
        """Calculate average iterations per task"""
        if not self.iterations_per_task:
            return 0.0
        return sum(self.iterations_per_task) / len(self.iterations_per_task)
    
    def get_duration(self) -> float:
        """Get total duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def print_report(self):
        """Print a comprehensive statistics report"""
        print("\n" + "="*80)
        print("PIPELINE STATISTICS REPORT")
        print("="*80)
        print(f"\n⏱️  Duration: {self.get_duration():.2f} seconds")
        print(f"\n Generation Summary:")
        print(f"   Total Attempts:     {self.total_attempts}")
        print(f"   ✅ Successful:      {self.successful_tasks} ({self.get_success_rate():.1f}%)")
        print(f"   ❌ Failed:          {self.failed_tasks}")
        print(f"   🔄 Avg Iterations:  {self.get_avg_iterations():.2f}")
        
        print(f"\n🔍 Validation Issues:")
        print(f"   Validation Failures:        {self.validation_failures}")
        print(f"   Committee Rejections:       {self.committee_rejections}")
        print(f"   User ID Inconsistencies:    {self.user_id_inconsistencies}")
        print(f"   Auto-corrections Applied:   {self.auto_corrections_applied}")
        print(f"   Generation Errors:          {self.generation_errors}")
        
        if self.failure_reasons:
            print(f"\n❌ Failure Breakdown:")
            for reason, count in sorted(self.failure_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {reason}: {count}")
        
        print("="*80 + "\n")


# Duplicate UserIDValidator removed — earlier definition retained at top of file.


class ScenarioDiversityChecker:
    """Checks and enforces scenario diversity across generated tasks"""
    
    # Scenario detection patterns based on primary actions
    SCENARIO_PATTERNS = TauBenchConfig().scenario_action_map
    
    @staticmethod
    def detect_scenario(task: Dict[str, Any]) -> Optional[str]:
        """Detect the primary scenario type of a task based on its actions"""
        if not isinstance(task, dict):
            return None
        
        actions = task.get('agt', [])
        if not actions:
            return None
        
        # Extract action names
        action_names = [a.get('name', '') for a in actions if isinstance(a, dict)]
        
        # Find matching scenario (check dominant actions)
        scenario_scores = defaultdict(int)
        for scenario, patterns in ScenarioDiversityChecker.SCENARIO_PATTERNS.items():
            for action_name in action_names:
                if action_name in patterns:
                    scenario_scores[scenario] += 1
        
        if not scenario_scores:
            return 'other'
        
        # Return scenario with highest score
        return max(scenario_scores.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def check_diversity(tasks: List[Dict[str, Any]], threshold: float = 0.5) -> Tuple[bool, Dict[str, int]]:
        """
        Check if tasks have sufficient scenario diversity.
        
        Args:
            tasks: List of tasks to check
            threshold: Max allowed proportion of any single scenario (0.5 = 50%)
            
        Returns:
            Tuple of (is_diverse, scenario_counts)
        """
        if not tasks:
            return True, {}
        
        scenario_counts = defaultdict(int)
        for task in tasks:
            scenario = ScenarioDiversityChecker.detect_scenario(task)
            if scenario:
                scenario_counts[scenario] += 1
        
        # Check if any scenario exceeds threshold
        total = len(tasks)
        max_proportion = max(scenario_counts.values()) / total if total > 0 else 0
        is_diverse = max_proportion <= threshold
        
        return is_diverse, dict(scenario_counts)
    
    @staticmethod
    def get_suggested_scenario(existing_tasks: List[Dict[str, Any]], available_scenarios: List[str]) -> Optional[str]:
        """
        Suggest a scenario to generate next based on what's already been generated.
        Prioritizes underrepresented scenarios.
        """
        if not existing_tasks:
            return available_scenarios[0] if available_scenarios else None
        
        # Count existing scenarios
        scenario_counts = defaultdict(int)
        for task in existing_tasks:
            scenario = ScenarioDiversityChecker.detect_scenario(task)
            if scenario:
                scenario_counts[scenario] += 1
        
        # Find least represented scenario from available list
        min_count = float('inf')
        suggested = None
        
        for scenario in available_scenarios:
            count = scenario_counts.get(scenario, 0)
            if count < min_count:
                min_count = count
                suggested = scenario
        
        return suggested


class UserIDValidator:
    """Validates user_id consistency across all actions in a task"""
    
    @staticmethod
    def validate_user_id_consistency(task: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if all actions use the same user_id consistently.
        
        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        if not isinstance(task, dict):
            return False, ["Task is not a dictionary"]
        
        actions = task.get('agt', [])
        if not isinstance(actions, list) or len(actions) == 0:
            return True, []  # No actions to validate
        
        user_ids_found = set()
        issues = []
        
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            
            args = action.get('arguments', {})
            if not isinstance(args, dict):
                continue
            
            user_id = args.get('user_id')
            if user_id:
                user_ids_found.add(user_id)
                
                # Check for inconsistency
                if len(user_ids_found) > 1:
                    issues.append(
                        f"Action {idx} ({action.get('name', 'unknown')}) uses "
                        f"user_id '{user_id}', but previous actions used: "
                        f"{user_ids_found - {user_id}}"
                    )
        
        is_consistent = len(user_ids_found) <= 1
        
        if not is_consistent:
            issues.insert(0, f"Found {len(user_ids_found)} different user_ids: {user_ids_found}")
        
        return is_consistent, issues
    
    @staticmethod
    def extract_canonical_user_id(task: Dict[str, Any]) -> Optional[str]:
        """
        Extract the most commonly used user_id from task actions.
        This is used as the "canonical" user_id for correction.
        """
        if not isinstance(task, dict):
            return None
        
        actions = task.get('agt', [])
        user_id_counts = defaultdict(int)
        
        for action in actions:
            if isinstance(action, dict):
                args = action.get('arguments', {})
                if isinstance(args, dict):
                    user_id = args.get('user_id')
                    if user_id:
                        user_id_counts[user_id] += 1
        
        if not user_id_counts:
            return None
        
        # Return most common user_id
        return max(user_id_counts.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def correct_user_id_inconsistencies(task: Dict[str, Any], canonical_user_id: str) -> int:
        """
        Correct all user_ids in actions to use the canonical user_id.
        
        Returns:
            Number of corrections made
        """
        if not isinstance(task, dict) or not canonical_user_id:
            return 0
        
        actions = task.get('agt', [])
        corrections = 0
        
        for action in actions:
            if isinstance(action, dict):
                args = action.get('arguments', {})
                if isinstance(args, dict) and 'user_id' in args:
                    if args['user_id'] != canonical_user_id:
                        args['user_id'] = canonical_user_id
                        corrections += 1
        
        return corrections


class ReviewCommittee:
    """A lightweight committee that uses the generator's API to simulate
    multiple reviewers. Enhanced with user_id consistency checking."""

    def __init__(self, size: int = 3, reviewers: Optional[List] = None, check_user_id: bool = True):
        self.size = size
        self.reviewers = reviewers or []
        self.check_user_id = check_user_id

    def review(self, task: Any) -> List[Dict[str, Any]]:
        """Return a list of review dicts: {accept: bool, score: float, comment: str}.
        Default reviewer: check coherence between q and agt lengths, presence of outputs,
        and user_id consistency.
        """
        reviews = []

        for i in range(self.size):
            # If custom reviewer callable provided, use it
            if i < len(self.reviewers) and callable(self.reviewers[i]):
                try:
                    r = self.reviewers[i](task)
                    reviews.append(r)
                    continue
                except Exception as e:
                    logger.warning(f"Reviewer {i} failed: {e}")

            # Default heuristic reviewer. Be defensive: `task` may not be a dict
            if isinstance(task, dict):
                q = task.get('q', '') or ''
                agt = task.get('agt', []) or []
                ogt = task.get('ogt', []) or []
            elif isinstance(task, list):
                # If a list of actions was passed as task, treat it as agt
                agt = task
                q = ''
                ogt = []
                # If first element seems to be a task dict, extract q/ogt
                if len(task) > 0 and isinstance(task[0], dict):
                    first = task[0]
                    if 'q' in first:
                        q = first.get('q', '')
                    if 'ogt' in first:
                        ogt = first.get('ogt', []) or []
            else:
                q = str(task)[:200]
                agt = []
                ogt = []

            score = 0.0
            comment_parts = []

            # Check instruction quality
            if q and len(q) > 40:
                score += 0.3
            else:
                comment_parts.append('short_instruction')

            # Check actions
            if agt and isinstance(agt, list) and len(agt) >= 1:
                score += 0.3
            else:
                comment_parts.append('missing_or_empty_actions')

            # Check outputs
            if ogt is not None:
                score += 0.2
            
            # Check user_id consistency if enabled
            if self.check_user_id and isinstance(task, dict):
                is_consistent, issues = UserIDValidator.validate_user_id_consistency(task)
                if is_consistent:
                    score += 0.2
                else:
                    comment_parts.append('user_id_inconsistent')
                    comment_parts.extend(issues[:2])  # Add first 2 issues

            accept = score >= 0.8
            comment = '; '.join(comment_parts) if comment_parts else 'ok'

            reviews.append({'accept': accept, 'score': score, 'comment': comment})

        return reviews


class FeedbackGenerator:
    """Aggregate validation errors and review comments, produce simple
    instructions for the generator to refine a task."""

    @staticmethod
    def generate_feedback(validation_report: Any, reviews: List[Dict[str, Any]]) -> str:
        parts = []
        # Validation report is TaskValidator.ValidationReport instance or dict-like
        if hasattr(validation_report, 'missing'):
            missing = getattr(validation_report, 'missing') or []
        else:
            missing = validation_report.get('missing', []) if isinstance(validation_report, dict) else []

        if missing:
            parts.append(f"fix_missing:{len(missing)}")

        # Summarize reviews
        rejects = [r for r in reviews if not r.get('accept')]
        if rejects:
            parts.append(f"review_rejections:{len(rejects)}")
            # include top comments
            comments = ','.join(set(r.get('comment','') for r in rejects))
            parts.append(f"comments:{comments}")

        if not parts:
            return "no_feedback"

        return ' | '.join(parts)


class TaskConfigurationPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.reader = TauBenchDataReader(config.envs_path)
        gen_config = TauBenchConfig()
        # Initialize generator with AgentFlow option if requested
        self.generator = TauBenchOpenAIGenerator(
            config.envs_path,
            use_agentflow=config.use_agentflow,
            agentflow_max_turns=config.agentflow_max_turns
        )
        self.validator = TaskValidator(self.reader)
        self.committee = ReviewCommittee(
            size=config.committee_size,
            check_user_id=config.enable_user_id_validation
        )
        self.stats = PipelineStatistics()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    def prepare_context(self) -> Dict[str, Any]:
        """Assemble context used by the generator: tools, policies, domain samples."""
        data = self.reader.generate_complete_prompt_data()
        return data

    def generate_initial(self, custom_user_id: Optional[str] = None, suggested_scenario: Optional[str] = None) -> Dict[str, Any]:
        """Generate a single task using the generator with real data.
        Enhanced with retry logic for generation failures and scenario guidance.
        
        Args:
            custom_user_id: Optional specific user ID
            suggested_scenario: Optional scenario hint for diversity (e.g., 'order_cancellation')
        """
        last_error = None
        
        for attempt in range(self.config.max_generation_retries + 1):
            try:
                # If pipeline requested AgentFlow, force that mode; otherwise use generator defaults
                force_mode = 'agentflow' if self.config.use_agentflow else None
                result = self.generator.generate_task_with_real_data(
                    custom_user_id=custom_user_id,
                    include_metadata=self.config.include_metadata,
                    suggested_scenario=suggested_scenario,
                    force_mode=force_mode
                )
                
                if result.get('success'):
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')
                    logger.warning(f"Generation attempt {attempt + 1} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Generation attempt {attempt + 1} raised exception: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.config.max_generation_retries:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # All retries exhausted
        self.stats.generation_errors += 1
        self.stats.add_failure_reason(f"generation_error: {last_error}")
        return {
            'success': False,
            'error': f"Failed after {self.config.max_generation_retries + 1} attempts: {last_error}"
        }

    def format_and_execute_check(self, task: Dict[str, Any]) -> Tuple[bool, Any]:
        """Use TaskValidator to check structural correctness and executability.
        Returns (is_valid, report)
        """
        report = self.validator.validate(task)
        is_valid = getattr(report, 'valid', False)
        
        if not is_valid:
            self.stats.validation_failures += 1
        
        return is_valid, report
    
    def check_user_id_consistency(self, task: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check and optionally correct user_id consistency"""
        if not self.config.enable_user_id_validation:
            return True, []
        
        is_consistent, issues = UserIDValidator.validate_user_id_consistency(task)
        
        if not is_consistent:
            self.stats.user_id_inconsistencies += 1
            logger.warning(f"User ID inconsistency detected: {issues[0] if issues else 'unknown'}")
            
            # Attempt auto-correction
            canonical_user_id = UserIDValidator.extract_canonical_user_id(task)
            if canonical_user_id:
                corrections = UserIDValidator.correct_user_id_inconsistencies(task, canonical_user_id)
                if corrections > 0:
                    logger.info(f"Auto-corrected {corrections} user_id inconsistencies to '{canonical_user_id}'")
                    self.stats.auto_corrections_applied += corrections
                    # Re-validate after correction
                    is_consistent, issues = UserIDValidator.validate_user_id_consistency(task)
        
        return is_consistent, issues

    def review_committee(self, task: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        reviews = self.committee.review(task)
        accepts = sum(1 for r in reviews if r.get('accept'))
        ok = accepts >= self.config.min_accepts
        
        if not ok:
            self.stats.committee_rejections += 1
        
        return ok, reviews

    def refine_loop(self, initial_task: Dict[str, Any], task_num: int) -> Dict[str, Any]:
        """Run the iterative refinement loop until the task validates or max_iterations reached.
        Enhanced with better logging and user_id consistency checks.
        """
        task = initial_task.copy()
        # If initial task contains suggested_scenario metadata and it's mismatched, try regenerating
        suggested = task.get('metadata', {}).get('suggested_scenario')
        matched = task.get('metadata', {}).get('suggested_scenario_matched')
        gen_config = getattr(self.generator, 'config', None) if hasattr(self, 'generator') else None
        strict = getattr(gen_config, 'scenario_match_strict', False) if gen_config else False
        if suggested and matched is False and strict:
            # If the pipeline wants strict scenario enforcement, try re-generating via generator
            max_retries = self.config.max_generation_retries
            for attempt in range(max_retries):
                logger.info(f"  ⚠️ Suggested scenario '{suggested}' mismatch - attempting re-generation ({attempt+1}/{max_retries})")
                new_res = self.generate_initial(suggested_scenario=suggested)
                if new_res.get('success'):
                    new_task = new_res.get('task', {})
                    if new_task.get('metadata', {}).get('suggested_scenario_matched'):
                        task = new_task
                        logger.info("  ✅ Regenerated task matches suggested scenario")
                        break
                    else:
                        task = new_task
                else:
                    logger.warning("  ⚠️ Regeneration for scenario match failed")
        iteration = 0

        logger.info(f"🔄 Starting refinement for task #{task_num}")
        
        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"  Iteration {iteration}/{self.config.max_iterations}")
            
            # Check user_id consistency first
            is_consistent, user_id_issues = self.check_user_id_consistency(task)
            if not is_consistent and user_id_issues:
                logger.warning(f"  ⚠️  User ID issues: {user_id_issues[0]}")
            
            # Validate format and executability
            is_valid, report = self.format_and_execute_check(task)

            if is_valid and is_consistent:
                logger.info("  ✅ Task passed validation checks")
                # Run committee
                ok, reviews = self.review_committee(task)
                if ok:
                    logger.info("  ✅ Task accepted by review committee")
                    task['_validated'] = True
                    task['_validation_report'] = self._serialize_report(report)
                    task['_reviews'] = reviews
                    task['_iterations'] = iteration
                    self.stats.iterations_per_task.append(iteration)
                    # Normalize action arguments before returning (ensure compatibility with tool signatures)
                    try:
                        from utils.arg_normalizer import normalize_action_arguments
                        for a in task.get('agt', []) or []:
                            name = a.get('name')
                            args = a.get('arguments', {}) or {}
                            try:
                                a['arguments'] = normalize_action_arguments(name, args)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return task
                else:
                    logger.info("  ❌ Task rejected by committee")
                    # Extract rejection reasons
                    rejection_comments = [r.get('comment', '') for r in reviews if not r.get('accept')]
                    logger.info(f"  Rejection reasons: {'; '.join(rejection_comments[:2])}")
                    
                    # Request a new sample
                    logger.info("  🔄 Requesting new generated sample...")
                    new_result = self.generate_initial()
                    if new_result.get('success'):
                        task = new_result.get('task', task)
                        continue
                    else:
                        logger.warning("  ⚠️  Generator failed; generating feedback")
                        feedback = FeedbackGenerator.generate_feedback(report, reviews)
                        logger.info(f"  Feedback: {feedback}")
            else:
                # Validation or consistency failed
                if not is_valid:
                    logger.info("  ❌ Task failed format/validation checks")
                if not is_consistent:
                    logger.info("  ❌ Task has user_id inconsistencies")
                
                ok, reviews = self.review_committee(task)
                feedback = FeedbackGenerator.generate_feedback(report, reviews)
                logger.info(f"  Feedback: {feedback}")

            # Try auto-corrections
            try:
                applied = self.validator.apply_suggestions(task, report)
                corrections = applied.get('corrections', []) if isinstance(applied, dict) else []
                if corrections:
                    logger.info(f" Applied {len(corrections)} auto-corrections")
                    self.stats.auto_corrections_applied += len(corrections)

                    # Re-validate the task after automatic corrections
                    report2 = self.validator.validate(task)
                    is_valid2 = report2.valid
                    is_consistent2 = True
                    if self.config.enable_user_id_validation:
                        # Check user_id consistency
                        is_consistent2, _ = UserIDValidator.validate_user_id_consistency(task)

                    if is_valid2 and is_consistent2:
                        logger.info("  ✅ Task validated after auto-corrections; accepting corrected task")
                        # Update working report and continue with accepted task
                        report = report2
                        is_valid = True
                        is_consistent = True
                        break  # accept task and stop refining
                    else:
                        logger.info("  ⚠️ Task still invalid after auto-corrections; requesting new sample")
                        new_result = self.generate_initial()
                        if new_result.get('success'):
                            task = new_result.get('task', task)
                            continue
                        else:
                            logger.warning("  ⚠️  Generator failed; stopping refinement")
                            self.stats.add_failure_reason("generator_exhausted")
                            break
                else:
                    # No corrections possible; request fresh sample
                    logger.info("  🔄 No auto-corrections available; requesting new sample")
                    new_result = self.generate_initial()
                    if new_result.get('success'):
                        task = new_result.get('task', task)
                    else:
                        logger.warning("  ⚠️  Generator failed; stopping refinement")
                        self.stats.add_failure_reason("generator_exhausted")
                        break
            except Exception as e:
                logger.error(f"  ❌ Error applying suggestions: {e}")
                self.stats.add_failure_reason(f"suggestion_error: {str(e)[:50]}")

        # Max iterations reached or failed
        logger.warning(f"  ❌ Task failed after {iteration} iterations")
        task['_validated'] = False
        task['_validation_report'] = self._serialize_report(report) if 'report' in locals() else {}
        task['_reviews'] = reviews if 'reviews' in locals() else []
        task['_iterations'] = iteration
        self.stats.iterations_per_task.append(iteration)
        self.stats.add_failure_reason("max_iterations_exceeded")
        return task

    def _serialize_report(self, report: Any) -> Dict[str, Any]:
        # Attempt to turn ValidationReport dataclass-like into dict
        try:
            return report.__dict__
        except Exception:
            return dict(report) if isinstance(report, dict) else {'valid': False}

    def _strip_internal_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of task with all keys starting with '_' removed.
        This keeps original task objects intact in memory while ensuring saved files
        don't contain internal bookkeeping fields.
        """
        if not isinstance(task, dict):
            return task
        return {k: v for k, v in task.items() if not (isinstance(k, str) and k.startswith('_'))}

    def run(self, num_tasks: int = 1) -> List[Dict[str, Any]]:
        """Run the complete pipeline for num_tasks tasks.
        Enhanced with progress tracking, comprehensive statistics, and scenario diversity enforcement.
        """
        results = []
        ctx = self.prepare_context()
        logger.info(f" Prepared context for generation")
        logger.info(f" Target: Generate {num_tasks} tasks")
        logger.info(f" Config: max_iterations={self.config.max_iterations}, "
                   f"committee_size={self.config.committee_size}, "
                   f"user_id_validation={self.config.enable_user_id_validation}, "
                   f"scenario_diversity={self.config.enforce_scenario_diversity}")
        print("\n" + "="*80)

        for i in range(num_tasks):
            self.stats.total_attempts += 1
            task_num = i + 1
            
            # Get suggested scenario for diversity
            suggested_scenario = None
            if self.config.enforce_scenario_diversity:
                suggested_scenario = ScenarioDiversityChecker.get_suggested_scenario(
                    results, 
                    self.config.scenario_rotation
                )
                if suggested_scenario:
                    logger.info(f"🎲 Suggested scenario for diversity: {suggested_scenario}")
            
            logger.info(f"\n Task {task_num}/{num_tasks} - Generating initial version...")
            
            # Pass scenario hint to generator
            res = self.generate_initial(suggested_scenario=suggested_scenario)
            
            if not res.get('success'):
                error_msg = res.get('error', 'Unknown error')
                logger.error(f"❌ Generator failed for task {task_num}: {error_msg}")
                self.stats.failed_tasks += 1
                self.stats.add_failure_reason(f"initial_generation_failed: {error_msg[:50]}")
                continue

            task = res.get('task', {})
            
            # Detect and log scenario
            detected_scenario = ScenarioDiversityChecker.detect_scenario(task)
            if detected_scenario:
                logger.info(f" Detected scenario: {detected_scenario}")
                task['_scenario'] = detected_scenario
            
            # Display task preview
            q_preview = task.get('q', '')[:100] + "..." if len(task.get('q', '')) > 100 else task.get('q', '')
            logger.info(f" Query preview: {q_preview}")
            
            refined = self.refine_loop(task, task_num)
            
            # Check final status
            if refined.get('_validated', False):
                logger.info(f"✅ Task {task_num} completed successfully")
                self.stats.successful_tasks += 1
            else:
                logger.warning(f"❌ Task {task_num} failed validation")
                self.stats.failed_tasks += 1
            
            results.append(refined)
            
            # Print progress with diversity info
            if self.config.enforce_scenario_diversity and len(results) > 0:
                is_diverse, scenario_counts = ScenarioDiversityChecker.check_diversity(results)
                diversity_status = "✅ Diverse" if is_diverse else "⚠️  Low diversity"
                print(f"\n Progress: {task_num}/{num_tasks} tasks processed "
                      f"({self.stats.successful_tasks} successful, {self.stats.failed_tasks} failed)")
                print(f"Scenario diversity: {diversity_status}")
                print(f"   Scenarios: {dict(scenario_counts)}")
            else:
                print(f"\n Progress: {task_num}/{num_tasks} tasks processed "
                      f"({self.stats.successful_tasks} successful, {self.stats.failed_tasks} failed)")
            print("="*80)

        # Finalize statistics
        self.stats.finalize()
        
        # Print diversity report
        if self.config.enforce_scenario_diversity:
            is_diverse, scenario_counts = ScenarioDiversityChecker.check_diversity(results)
            print(f"\n🎨 Final Scenario Diversity Report:")
            print(f"   Status: {'✅ Diverse' if is_diverse else '⚠️  Needs improvement'}")
            print(f"   Distribution: {dict(scenario_counts)}")
            print()
        
        # Strip internal underscore-prefixed metadata before saving
        cleaned = [self._strip_internal_metadata(t) for t in results]

        # Normalize action arguments to match tool signatures before writing
        try:
            from utils.arg_normalizer import normalize_action_arguments
            for task in cleaned:
                agt = task.get('agt', []) or []
                for a in agt:
                    name = a.get('name')
                    args = a.get('arguments', {}) or {}
                    try:
                        a['arguments'] = normalize_action_arguments(name, args)
                    except Exception:
                        pass
        except Exception:
            logger.debug("Could not import argument normalizer; skipping normalization before save")

        # Save results
        try:
            with open(self.config.save_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(cleaned)} tasks to {self.config.save_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            self.stats.add_failure_reason(f"save_error: {str(e)[:50]}")

        # Print comprehensive statistics report
        self.stats.print_report()

        return results


def main():
    """Main entry point with configurable parameters"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task Configuration Pipeline')
    parser.add_argument('--num-tasks', type=int, default=TauBenchConfig.num_tasks, help='Number of tasks to generate')
    parser.add_argument('--max-iterations', type=int, default=3, help='Max refinement iterations per task')
    parser.add_argument('--output', type=str, default='generated_tasks/Sampled_Tasks.json', 
                       help='Output file path')
    parser.add_argument('--no-user-id-validation', action='store_true', 
                       help='Disable user_id consistency validation')
    parser.add_argument('--committee-size', type=int, default=3, help='Review committee size')
    parser.add_argument('--agentflow', action='store_true', help='Use AgentFlow multi-turn generation')
    parser.add_argument('--agentflow-turns', type=int, default=5, help='Max turns for AgentFlow generation')
    
    args = parser.parse_args()
    
    cfg = PipelineConfig(
        max_iterations=args.max_iterations,
        save_path=args.output,
        enable_user_id_validation=not args.no_user_id_validation,
        committee_size=args.committee_size
    )
    # Apply AgentFlow CLI options
    cfg.use_agentflow = args.agentflow
    cfg.agentflow_max_turns = args.agentflow_turns
    
    print("\n  Starting Task Generation Pipeline")
    print(f"   Target tasks: {args.num_tasks}")
    print(f"   Output: {args.output}")
    print(f"   User ID validation: {'enabled' if cfg.enable_user_id_validation else 'disabled'}")
    print()
    
    pipeline = TaskConfigurationPipeline(cfg)
    tasks = pipeline.run(num_tasks=args.num_tasks)
    
    successful = sum(1 for t in tasks if t.get('_validated', False))
    print(f"\n✅ Pipeline completed: {successful}/{len(tasks)} tasks validated successfully")


if __name__ == '__main__':
    main()

