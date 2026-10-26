"""Causal feature definitions and frozen standard service units."""
import json, math, re

def log(x):
    return math.log1p(max(0, x))


def issue_text(messages):
    text = next((m.get('content') or '' for m in messages if m['role']=='user'), '')
    match = re.search(r'<pr_description>(.*?)</pr_description>', text, re.S)
    return match.group(1).strip() if match else text[:12000]


def commands(messages):
    result=[]
    for m in messages:
        if m['role'] != 'assistant':
            continue
        for t in m.get('tool_calls') or []:
            try:
                obj=json.loads(t.get('function',{}).get('arguments','{}'))
            except (ValueError,TypeError):
                continue
            if isinstance(obj,dict) and isinstance(obj.get('command'),str):
                result.append(obj['command'])
        # Some original agents record actions but no OpenAI tool_calls.
        if not m.get('tool_calls'):
            result.extend(r['command'] for r in m.get('extra',{}).get('actions',[])
                          if isinstance(r,dict) and isinstance(r.get('command'),str))
    return result


ACTION_PATTERNS = {
    'search':r'\b(grep|rg|find|ls)\b',
    'read':r'\b(cat|head|tail|sed|less)\b',
    'edit':r'(write_text|open\(.+?[wa][\x27\"]|sed\s+-i|apply_patch|replace\()',
    'test':r'\b(pytest|unittest|tox|runtests|test_)|reproduce',
    'diff':r'git\s+(diff|status)',
    'submit':r'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',
    'python':r'\bpython[23]?\b',
}


def causal_features(messages, prompt_lengths, outputs, services, after):
    """No metadata except currently observed lengths and completed messages."""
    text=issue_text(messages)
    cmds=commands(messages)
    feedback=[m.get('content') or '' for m in messages if m['role'] in ('tool','user')]
    # The initial issue is task context, not a tool result.
    results=[s for s in feedback if '<returncode>' in s or 'Tool call error:' in s]
    codes=[int(v) for s in results for v in re.findall(r'<returncode>(-?\d+)</returncode>',s)]
    errors=[s for s in results if 'Tool call error:' in s or '<exception>' in s]
    last=results[-1] if results else ''
    growth=[b-a for a,b in zip(prompt_lengths,prompt_lengths[1:])]
    vals={
        'after':int(after), 'call_index':len(prompt_lengths)-1,
        'prompt_log':log(prompt_lengths[-1]), 'initial_prompt_log':log(prompt_lengths[0]),
        'outputs_seen':len(outputs), 'output_sum_log':log(sum(outputs)),
        'output_last_log':log(outputs[-1] if outputs else 0),
        'output_mean_log':log(sum(outputs)/len(outputs) if outputs else 0),
        'output_max_log':log(max(outputs,default=0)),
        'growth_last_log':log(growth[-1] if growth else 0),
        'growth_mean_log':log(sum(growth)/len(growth) if growth else 0),
        'service_seen_log':log(sum(services)),
        'issue_chars_log':log(len(text)), 'issue_lines_log':log(text.count('\n')),
        'issue_code_fraction':sum(c in '`{}[]()=_' for c in text)/max(1,len(text)),
        'issue_links_log':log(len(re.findall(r'https?://',text))),
        'issue_error_words_log':log(len(re.findall(r'error|exception|traceback|fail',text,re.I))),
        'commands_log':log(len(cmds)), 'unique_commands_log':log(len(set(cmds))),
        'repeat_commands_log':log(len(cmds)-len(set(cmds))),
        'feedback_count_log':log(len(results)), 'nonzero_exits_log':log(sum(c!=0 for c in codes)),
        'zero_exits_log':log(sum(c==0 for c in codes)), 'tool_errors_log':log(len(errors)),
        'last_feedback_chars_log':log(len(last)),
        'last_nonzero':int(bool(codes) and codes[-1]!=0),
        'last_test_failed':int(bool(re.search(r'\b(FAILED|ERRORS?|[1-9]\d* failed)\b',last))),
        'last_test_passed':int(bool(re.search(r'\b[1-9]\d* passed\b',last))),
        'last_traceback':int('Traceback (most recent call last)' in last),
        'repeat_feedback_log':log(len(results)-len(set(results))),
        'submit_visible':int(any('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in c for c in cmds)),
    }
    for label,pattern in ACTION_PATTERNS.items():
        vals[label+'_total_log']=log(sum(bool(re.search(pattern,c,re.I)) for c in cmds))
        vals[label+'_recent_log']=log(sum(bool(re.search(pattern,c,re.I)) for c in cmds[-3:]))
    # Only paths typed in already observed commands; no environment/image metadata.
    paths=set(re.findall(r'[\w./-]+\.py\b','\n'.join(cmds)))
    vals['observed_py_paths_log']=log(len(paths))
    return vals,text


def lcp(a,b):
    for j,(x,y) in enumerate(zip(a,b)):
        if x!=y:
            return j
    return min(len(a),len(b))


def service(n,o,prefix,profile):
    b=profile['beta'];q=n-prefix
    # Chunked prefill: each nonempty chunk pays the calibrated fixed overhead.
    prefill=0.;cursor=prefix
    while cursor<n:
        q=min(2048,n-cursor)
        prefill+=max(0.,b[0]+b[1]*q+b[2]*q*(2*cursor+q+1)/2+b[3])
        cursor+=q
    return prefill+o*profile['decode_step_by_seqs']['1']


