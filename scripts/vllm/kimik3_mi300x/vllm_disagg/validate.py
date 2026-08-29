import json,urllib.request,time
URL="http://10.32.82.3:30000"
def test(label, body_extra):
    body=json.dumps({"model":"kimi-k3","messages":[{"role":"user","content":"The secret code is 8241. What is the secret code? Answer with only the number."}],"max_tokens":128,"temperature":0,**body_extra}).encode()
    t=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=120)
        j=json.loads(r.read());m=j["choices"][0]["message"]
        c=m.get("content") or ""; rc=m.get("reasoning_content") or ""
        print(f"[{label}] {time.time()-t:.1f}s finish={j['choices'][0]['finish_reason']} recall={'8241' in (c+rc)}")
        print(f"   content={repr(c[:90])}")
        print(f"   reasoning={repr(rc[:90])}")
    except Exception as e: print(f"[{label}] ERR {str(e)[:70]}")
test("thinking=OFF", {"chat_template_kwargs":{"thinking":False}})
time.sleep(2)
test("thinking=ON ", {"chat_template_kwargs":{"thinking":True}})
