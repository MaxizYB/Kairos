"""Kairos features implementation."""
import math
import re

WORDS = (
    'error', 'traceback', 'exception', 'fail', 'passed', 'test', 'assert',
    'patch', 'diff', 'fix', 'bug', 'regression', 'import', 'class', 'function',
    'expected', 'actual', 'reproduce', 'warning', 'deprecated', 'performance',
    'documentation', 'feature', 'support', 'version', 'search', 'find',
    'pytest', 'unittest', 'git', 'python', 'sed', 'grep', 'submit',
)
FEATURE_NAMES = ('stage', 'prompt_log', 'initial_prompt_log', 'growth_log',
                 'output_mean_log', 'output_last_log', 'output_sum_log',
                 'growth_mean_log', 'growth_max_log', 'tail_chars_log',
                 'tail_lines_log', 'tail_digits_fraction', 'tail_code_fraction') + tuple(
                     f'{part}_{word}' for part in ('issue', 'tail') for word in WORDS)


def features(text, prompts, outputs):
    """Only the currently supplied prompt and completed length history enter.

    The fixed first/last windows bound parsing. Current assistant output,
    tool type for the not-yet-executed call, and completion flags are absent.
    """
    issue=text[:8192].lower();tail=text[-8192:].lower()
    growth=[max(0,b-a) for a,b in zip(prompts,prompts[1:])]
    log=lambda v: math.log1p(max(0,v))
    values=[len(outputs),log(prompts[-1]),log(prompts[0]),
            log(growth[-1] if growth else 0),
            log(sum(outputs)/len(outputs) if outputs else 0),
            log(outputs[-1] if outputs else 0),log(sum(outputs)),
            log(sum(growth)/len(growth) if growth else 0),log(max(growth,default=0)),
            log(len(tail)),log(tail.count('\n')),
            sum(c.isdigit() for c in tail)/max(1,len(tail)),
            sum(c in '{}[]()_`=:' for c in tail)/max(1,len(tail))]
    for part in (issue,tail):
        counts={word:0 for word in WORDS}
        for word in re.findall(r'[a-z]+',part):
            if word in counts:counts[word]+=1
        values.extend(log(counts[word]) for word in WORDS)
    return values
