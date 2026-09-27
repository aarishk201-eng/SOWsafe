import re

with open('main.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add UploadFile to imports
code = re.sub(r'from fastapi import FastAPI, HTTPException, Request', 'from fastapi import FastAPI, HTTPException, Request, UploadFile', code)

# 2. Fix hasattr(X, 'filename') to isinstance(X, UploadFile)
code = re.sub(r'([a-zA-Z0-9_]+) and hasattr\(\1, "filename"\)', r'isinstance(\1, UploadFile)', code)
code = re.sub(r'hasattr\(item, "filename"\) and hasattr\(item, "read"\)', r'isinstance(item, UploadFile)', code)

# 3. Cast form.get() to str where appropriate
code = re.sub(r'target_path = form\.get\("path"\)', 'target_path = str(form.get("path")) if form.get("path") else None', code)
code = re.sub(r'path_a = form\.get\("path_a"\)', 'path_a = str(form.get("path_a")) if form.get("path_a") else None', code)
code = re.sub(r'path_b = form\.get\("path_b"\)', 'path_b = str(form.get("path_b")) if form.get("path_b") else None', code)
code = re.sub(r'path_1 = form\.get\("path_1"\)', 'path_1 = str(form.get("path_1")) if form.get("path_1") else None', code)
code = re.sub(r'path_2 = form\.get\("path_2"\)', 'path_2 = str(form.get("path_2")) if form.get("path_2") else None', code)
code = re.sub(r't1_path = form\.get\("t1_path"\)', 't1_path = str(form.get("t1_path")) if form.get("t1_path") else None', code)
code = re.sub(r't2_path = form\.get\("t2_path"\)', 't2_path = str(form.get("t2_path")) if form.get("t2_path") else None', code)
code = re.sub(r't1 = form\.get\("t1"\)', 't1 = str(form.get("t1")) if form.get("t1") else None', code)
code = re.sub(r't2 = form\.get\("t2"\)', 't2 = str(form.get("t2")) if form.get("t2") else None', code)
code = re.sub(r'path_opt = form\.get\("path_opt"\)', 'path_opt = str(form.get("path_opt")) if form.get("path_opt") else None', code)
code = re.sub(r'path_sar = form\.get\("path_sar"\)', 'path_sar = str(form.get("path_sar")) if form.get("path_sar") else None', code)
code = re.sub(r'path_opt = form\.get\("optical"\)', 'path_opt = str(form.get("optical")) if form.get("optical") else None', code)
code = re.sub(r'path_sar = form\.get\("sar"\)', 'path_sar = str(form.get("sar")) if form.get("sar") else None', code)

# 4. Fix _query / q assignments
code = re.sub(r'_query = form\.get\("query"\)', '_query = str(form.get("query")) if form.get("query") else None', code)
code = re.sub(r'_query = form\.get\("question"\)', '_query = str(form.get("question")) if form.get("question") else None', code)
code = re.sub(r'_phrase = form\.get\("phrase"\)', '_phrase = str(form.get("phrase")) if form.get("phrase") else None', code)

# 5. Fix form lists
code = re.sub(r'q = form\.get\("query"\)', 'q = str(form.get("query")) if form.get("query") else None', code)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(code)
