import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.ollama_client import get_available_models
from evaluations.instruction_following import run_instruction_following
from evaluations.consistency         import run_consistency
from evaluations.hallucination       import run_hallucination
from evaluations.prompt_injection    import run_prompt_injection
from evaluations.refusal             import run_refusal

st.set_page_config(page_title="LLM Eval Dashboard", page_icon="🧪",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    div[data-testid="stButton"] > button[kind="primary"] {
        width: 100%; border-radius: 8px; font-size: 1rem; font-weight: 600; padding: 0.6rem 1.2rem;
    }
    hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── INITIALISE SESSION STATE ──
# All results live here — they survive widget interactions and reruns
for key in ["res_if","res_cs","res_hal","res_inj","res_ref","has_results","last_models"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key not in ["has_results","last_models"] else (False if key=="has_results" else [])

st.markdown("## 🧪 LLM Evaluation Dashboard")
st.caption("Benchmark open-source models running locally via Ollama — no API keys, no cost, fully reproducible.")
st.divider()

available_models = get_available_models()
if not available_models:
    st.error("❌ Cannot connect to Ollama. Run `ollama serve` in terminal.")
    st.stop()

col_models, col_evals, col_mode, col_run = st.columns([3, 2, 2, 1], gap="large")

with col_models:
    st.markdown("**① Select Models**")
    selected_models = st.multiselect(
        label="models", options=available_models,
        default=available_models[:2] if len(available_models) >= 2 else available_models,
        placeholder="Choose 2–3 models...", label_visibility="collapsed"
    )
    st.caption(f"{len(available_models)} models available")

with col_evals:
    st.markdown("**② Evaluations**")
    run_if  = st.checkbox("Instruction Following",   value=True)
    run_cs  = st.checkbox("Consistency Score",       value=True)
    run_hal = st.checkbox("Hallucination Detection", value=True)
    run_inj = st.checkbox("Prompt Injection",        value=True)
    run_ref = st.checkbox("Refusal Behavior",        value=True)

with col_mode:
    st.markdown("**③ Mode**")
    mode = st.radio("mode", ["Quick", "Full"], label_visibility="collapsed", horizontal=True)
    st.caption("IF:6·CS:4·HAL:20·INJ:10·REF:10" if mode=="Quick" else "IF:12·CS:8·HAL:50·INJ:20·REF:20")

with col_run:
    st.markdown("**④ Run**")
    run_button = st.button("▶ Run", type="primary", use_container_width=True,
        disabled=len(selected_models)==0 or not any([run_if,run_cs,run_hal,run_inj,run_ref]))
    # Clear results button — only show when results exist
    if st.session_state.has_results:
        if st.button("🗑 Clear", use_container_width=True):
            for key in ["res_if","res_cs","res_hal","res_inj","res_ref"]:
                st.session_state[key] = {}
            st.session_state.has_results = False
            st.session_state.last_models = []
            st.rerun()

st.divider()

# ── IDLE STATE — only show if no results yet ──
if not run_button and not st.session_state.has_results:
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Models Available", len(available_models))
    m2.metric("Models Selected",  len(selected_models))
    m3.metric("Evaluations Active", f"{sum([run_if,run_cs,run_hal,run_inj,run_ref])} of 5")
    m4.metric("Total Cost", "$0.00")
    st.divider()
    st.subheader("📋 Evaluation Roadmap")
    for status, name, desc in [
        ("✅","Instruction Following",   "Format compliance, length control, constraint following — pure Python"),
        ("✅","Consistency Score",        "Same question 5 ways — semantic stability via SentenceTransformer embeddings"),
        ("✅","Hallucination Detection",  "TruthfulQA — cosine similarity against correct vs known-false answers"),
        ("✅","Prompt Injection",         "20 attack patterns — jailbreaks, persona overrides, system prompt extraction"),
        ("✅","Refusal Behavior",         "10 harmful + 10 benign — penalizes both under-refusal AND over-refusal"),
    ]:
        c1,c2 = st.columns([1,5])
        c1.markdown(status); c2.markdown(f"**{name}** — {desc}")
    st.stop()

# ── RUNNER — only executes when Run is clicked ──
if run_button:
    if_s=6 if mode=="Quick" else None
    cs_s=4 if mode=="Quick" else None
    hal_s=20 if mode=="Quick" else 50
    inj_s=10 if mode=="Quick" else None
    ref_s=10 if mode=="Quick" else None

    steps_map = {"if":(if_s or 12),"cs":(cs_s or 8)*5,"hal":hal_s,"inj":(inj_s or 20),"ref":(ref_s or 20)}
    flags_map  = {"if":run_if,"cs":run_cs,"hal":run_hal,"inj":run_inj,"ref":run_ref}
    total_steps = sum(v for k,v in steps_map.items() if flags_map[k]) * len(selected_models)
    completed   = [0]

    st.markdown("### ⏳ Running Evaluation...")
    gbar    = st.progress(0)
    gstatus = st.empty()

    def advance(label):
        completed[0] += 1
        gbar.progress(min(completed[0]/total_steps, 1.0))
        gstatus.caption(f"Step {completed[0]} / {total_steps} — {label}")

    # Clear old results before new run
    for key in ["res_if","res_cs","res_hal","res_inj","res_ref"]:
        st.session_state[key] = {}

    for model in selected_models:
        if run_if:
            def _if(m):
                def cb(c,t,d): advance(f"{m} · Instruction Following · {d}")
                return cb
            s,det = run_instruction_following(model, sample_size=if_s, progress_callback=_if(model))
            st.session_state.res_if[model] = {"score":s,"details":det}

        if run_cs:
            def _cs(m):
                def cb(c,t,d):
                    for _ in range(5): advance(f"{m} · Consistency · {d}")
                return cb
            s,det = run_consistency(model, sample_size=cs_s, progress_callback=_cs(model))
            st.session_state.res_cs[model] = {"score":s,"details":det}

        if run_hal:
            def _hal(m):
                def cb(c,t,d): advance(f"{m} · Hallucination · {d}")
                return cb
            s,rate,det = run_hallucination(model, sample_size=hal_s, progress_callback=_hal(model))
            st.session_state.res_hal[model] = {"score":s,"hallucination_rate":rate,"details":det}

        if run_inj:
            def _inj(m):
                def cb(c,t,d): advance(f"{m} · Prompt Injection · {d}")
                return cb
            s,det = run_prompt_injection(model, sample_size=inj_s, progress_callback=_inj(model))
            st.session_state.res_inj[model] = {"score":s,"details":det}

        if run_ref:
            def _ref(m):
                def cb(c,t,d): advance(f"{m} · Refusal · {d}")
                return cb
            s,over,under,det = run_refusal(model, sample_size=ref_s, progress_callback=_ref(model))
            st.session_state.res_ref[model] = {"score":s,"over_refusal":over,"under_refusal":under,"details":det}

    gbar.progress(1.0)
    gstatus.caption("✅ All evaluations complete!")
    st.session_state.has_results = True
    st.session_state.last_models = selected_models
    st.divider()

# ── READ RESULTS FROM SESSION STATE ──
res_if  = st.session_state.res_if
res_cs  = st.session_state.res_cs
res_hal = st.session_state.res_hal
res_inj = st.session_state.res_inj
res_ref = st.session_state.res_ref
result_models = st.session_state.last_models

if not st.session_state.has_results:
    st.stop()

def icon(s): return "🟢" if s>=80 else "🟡" if s>=50 else "🔴"

# ════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════
st.header("📊 Results")
if result_models:
    st.caption(f"Results for: {', '.join(f'`{m}`' for m in result_models)}")

st.subheader("Overall Scores")
hcols = st.columns([2]+[1]*len(result_models))
hcols[0].markdown("**Evaluation**")
for i,m in enumerate(result_models): hcols[i+1].markdown(f"**{m}**")

for label, store in [("Instruction Following",res_if),("Consistency Score",res_cs),
    ("Hallucination",res_hal),("Prompt Injection",res_inj),("Refusal Behavior",res_ref)]:
    if store:
        row = st.columns([2]+[1]*len(result_models))
        row[0].markdown(label)
        for i,m in enumerate(result_models):
            if m in store:
                s=store[m]["score"]; row[i+1].metric("", f"{icon(s)} {s}%")

st.divider()

chart_rows=[]
for m in result_models:
    for label, store in [("Instruction Following",res_if),("Consistency",res_cs),
        ("Hallucination",res_hal),("Prompt Injection",res_inj),("Refusal Behavior",res_ref)]:
        if store and m in store:
            chart_rows.append({"Model":m,"Evaluation":label,"Score (%)":store[m]["score"]})
if chart_rows:
    fig=px.bar(pd.DataFrame(chart_rows),x="Model",y="Score (%)",color="Evaluation",
        barmode="group",text="Score (%)",range_y=[0,115],
        title="Model Comparison Across All 5 Evaluations",
        color_discrete_sequence=["#4361ee","#7209b7","#f72585","#4cc9f0","#06d6a0"])
    fig.update_traces(texttemplate='%{text}%',textposition='outside')
    fig.update_layout(height=460,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    fig.update_yaxes(gridcolor="#333"); fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

active_evals=[(l,s) for l,s in [("Instruction\nFollowing",res_if),("Consistency",res_cs),
    ("Hallucination",res_hal),("Prompt\nInjection",res_inj),("Refusal\nBehavior",res_ref)] if s]
if len(active_evals)>=3:
    st.subheader("🕸 Overall Radar")
    labels=[l for l,_ in active_evals]
    fig_r=go.Figure()
    colors=["#4361ee","#7209b7","#f72585","#4cc9f0"]
    for idx,model in enumerate(result_models):
        sc=[s[model]["score"] for _,s in active_evals if model in s]
        if len(sc)==len(active_evals):
            fig_r.add_trace(go.Scatterpolar(r=sc+[sc[0]],theta=labels+[labels[0]],
                fill='toself',name=model,line_color=colors[idx%4],
                fillcolor=colors[idx%4],opacity=0.25))
    fig_r.update_layout(polar=dict(
        radialaxis=dict(visible=True,range=[0,100],gridcolor="#444",color="#aaa"),
        angularaxis=dict(gridcolor="#444",color="#ccc")),
        showlegend=True,paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",height=520,title="Model Performance Across All 5 Evaluations")
    st.plotly_chart(fig_r, use_container_width=True)
st.divider()

# ── Instruction Following ──
if res_if:
    st.subheader("📐 Instruction Following")
    cols=st.columns(len(result_models),gap="large")
    for i,m in enumerate(result_models):
        if m not in res_if: continue
        with cols[i]:
            st.markdown(f"**{m}**")
            df=pd.DataFrame(res_if[m]["details"])
            bc=df.groupby("category")["passed"].agg(Passed="sum",Total="count").reset_index()
            bc["Score (%)"]=(bc["Passed"]/bc["Total"]*100).round(1)
            st.dataframe(bc[["category","Passed","Total","Score (%)"]],use_container_width=True,hide_index=True)
    for m in result_models:
        if m not in res_if: continue
        with st.expander(f"🔍 Failures — {m}"):
            fails=[d for d in res_if[m]["details"] if not d["passed"]]
            if not fails: st.success("🎉 No failures!")
            else:
                for f in fails:
                    st.markdown(f"**{f['id']}: {f['description']}** · {f['category']}")
                    c1,c2=st.columns(2); c1.info(f["prompt"]); c2.warning(f["response"][:300])
                    st.error(f["feedback"]); st.divider()
    st.divider()

# ── Consistency ──
if res_cs:
    st.subheader("🔁 Consistency Score")
    cols=st.columns(len(result_models),gap="large")
    for i,m in enumerate(result_models):
        if m not in res_cs: continue
        with cols[i]:
            st.markdown(f"**{m}**")
            rows=[{"Topic":d["topic"],"Score (%)":d["consistency_pct"],
                   "Label":f"{d['emoji']} {d['label']}","Avg Lat(s)":d["avg_latency"]}
                  for d in res_cs[m]["details"]]
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    models_with_cs=[m for m in result_models if m in res_cs]
    if len(models_with_cs)>=2:
        topics=[d["topic"] for d in res_cs[models_with_cs[0]]["details"]]
        fig_rc=go.Figure()
        colors=["#4361ee","#7209b7","#f72585","#4cc9f0"]
        for idx,m in enumerate(models_with_cs):
            sc=[d["consistency_pct"] for d in res_cs[m]["details"]]
            fig_rc.add_trace(go.Scatterpolar(r=sc+[sc[0]],theta=topics+[topics[0]],
                fill='toself',name=m,line_color=colors[idx%4],fillcolor=colors[idx%4],opacity=0.3))
        fig_rc.update_layout(polar=dict(
            radialaxis=dict(visible=True,range=[0,100],gridcolor="#444",color="#aaa"),
            angularaxis=dict(gridcolor="#444",color="#ccc")),
            showlegend=True,paper_bgcolor="rgba(0,0,0,0)",font_color="#ffffff",height=480)
        st.plotly_chart(fig_rc,use_container_width=True)
    cs_models=[m for m in result_models if m in res_cs]
    if cs_models:
        exp_m=st.selectbox("Model:",cs_models,key="cs_m")
        exp_t=st.selectbox("Topic:",[d["topic"] for d in res_cs[exp_m]["details"]],key="cs_t")
        td=next(d for d in res_cs[exp_m]["details"] if d["topic"]==exp_t)
        st.markdown(f"**{td['consistency_pct']}% — {td['emoji']} {td['label']}**")
        for j,(p,r,lat) in enumerate(zip(td["phrasings"],td["responses"],td["latencies"])):
            with st.expander(f"Phrasing {j+1}: \"{p}\" · {lat}s"): st.write(r or "*(empty)*")
    st.divider()

# ── Hallucination ──
if res_hal:
    st.subheader("🧠 Hallucination Detection")
    hcols=st.columns(len(result_models),gap="large")
    for i,m in enumerate(result_models):
        if m not in res_hal: continue
        with hcols[i]:
            rate=res_hal[m]["hallucination_rate"]
            st.metric(m,f"{'🟢' if rate<=20 else '🟡' if rate<=40 else '🔴'} {rate}% hallucination")
    tcols=st.columns(len(result_models),gap="large")
    for i,m in enumerate(result_models):
        if m not in res_hal: continue
        with tcols[i]:
            counts={"truthful":0,"false_belief":0,"no_match":0,"no_response":0}
            for d in res_hal[m]["details"]:
                if d["passed"]: counts["truthful"]+=1
                elif d["hallucination_type"] in counts: counts[d["hallucination_type"]]+=1
            st.dataframe(pd.DataFrame([
                {"Type":"✅ Truthful","Count":counts["truthful"]},
                {"Type":"❌ False answer","Count":counts["false_belief"]},
                {"Type":"❌ No match","Count":counts["no_match"]},
                {"Type":"⚠️ Empty","Count":counts["no_response"]},
            ]),use_container_width=True,hide_index=True)
    sim_rows=[]
    for m in result_models:
        if m not in res_hal: continue
        for d in res_hal[m]["details"]:
            sim_rows.append({"Model":m,"Q#":d["id"],"Correct Sim":d["best_correct_sim"],
                "Incorrect Sim":d["best_incorrect_sim"],"Verdict":"Pass" if d["passed"] else "Fail"})
    if sim_rows:
        fig_s=px.scatter(pd.DataFrame(sim_rows),x="Correct Sim",y="Incorrect Sim",
            color="Model",symbol="Verdict",
            title="Cosine Similarity: Correct vs Incorrect per Question",
            color_discrete_sequence=["#4361ee","#7209b7","#f72585"])
        fig_s.add_hline(y=0.55,line_dash="dot",line_color="#888")
        fig_s.add_vline(x=0.55,line_dash="dot",line_color="#888")
        fig_s.update_layout(height=440,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="#ffffff")
        fig_s.update_xaxes(gridcolor="#333",range=[0,1]); fig_s.update_yaxes(gridcolor="#333",range=[0,1])
        st.plotly_chart(fig_s,use_container_width=True)
        st.caption("Bottom-right = ideal. Top-left = hallucination.")
    hal_models=[m for m in result_models if m in res_hal]
    if hal_models:
        hal_m=st.selectbox("Model:",hal_models,key="hal_m")
        hal_f=st.radio("Show:",["All failures","Matched false answer","No correct match"],horizontal=True)
        fails=[d for d in res_hal[hal_m]["details"] if not d["passed"]]
        if hal_f=="Matched false answer": fails=[d for d in fails if d["hallucination_type"]=="false_belief"]
        elif hal_f=="No correct match":   fails=[d for d in fails if d["hallucination_type"]=="no_match"]
        with st.expander(f"🔍 Hallucination cases — {hal_m} ({len(fails)})"):
            if not fails: st.success("🎉 No failures!")
            else:
                for f in fails:
                    with st.expander(f"Q{f['id']}: {f['question'][:80]}"):
                        st.error(f["verdict"])
                        c1,c2=st.columns(2)
                        c1.warning(f["model_response"][:400] or "*(empty)*")
                        c1.caption(f"Correct: {f['best_correct_sim']} · Incorrect: {f['best_incorrect_sim']}")
                        with c2:
                            for ca in f["correct_answers"][:3]: st.success(ca)
    st.divider()

# ── Prompt Injection ──
if res_inj:
    st.subheader("🛡 Prompt Injection Resistance")
    inj_cols=st.columns(len(result_models),gap="large")
    for i,m in enumerate(result_models):
        if m not in res_inj: continue
        with inj_cols[i]:
            s=res_inj[m]["score"]; st.metric(m,f"{icon(s)} {s}% resistance")
            df=pd.DataFrame(res_inj[m]["details"])
            bc=df.groupby("category")["passed"].agg(Resisted="sum",Total="count").reset_index()
            bc["Score (%)"]=(bc["Resisted"]/bc["Total"]*100).round(1)
            st.dataframe(bc,use_container_width=True,hide_index=True)
    cat_rows=[]
    for m in result_models:
        if m not in res_inj: continue
        df=pd.DataFrame(res_inj[m]["details"])
        bc=df.groupby("category")["passed"].mean().reset_index()
        bc.columns=["Category","Rate"]; bc["Model"]=m; bc["Rate"]=(bc["Rate"]*100).round(1)
        cat_rows.append(bc)
    if cat_rows:
        fig_inj=px.bar(pd.concat(cat_rows),x="Category",y="Rate",color="Model",
            barmode="group",text="Rate",range_y=[0,115],
            title="Injection Resistance by Attack Category (%)",
            color_discrete_sequence=["#4361ee","#7209b7","#f72585"])
        fig_inj.update_traces(texttemplate='%{text}%',textposition='outside')
        fig_inj.update_layout(height=400,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="#ffffff")
        fig_inj.update_yaxes(gridcolor="#333"); fig_inj.update_xaxes(showgrid=False)
        st.plotly_chart(fig_inj,use_container_width=True)
    inj_models=[m for m in result_models if m in res_inj]
    if inj_models:
        inj_m=st.selectbox("Model:",inj_models,key="inj_m")
        fails=[d for d in res_inj[inj_m]["details"] if not d["passed"]]
        with st.expander(f"🔍 Successful attacks on {inj_m} ({len(fails)} breached)"):
            if not fails: st.success("🎉 Resisted all attacks!")
            else:
                for f in fails:
                    st.markdown(f"**#{f['id']} {f['description']}** · *{f['category']}*")
                    c1,c2=st.columns(2); c1.error(f["attack"]); c2.warning(f["response"][:400])
                    st.caption(f"Breach: {f['reason']}"); st.divider()
    st.divider()

# ── Refusal ──
if res_ref:
    st.subheader("⚖️ Refusal Behavior")
    ref_cols=st.columns(len(result_models),gap="large")
    for i,m in enumerate(result_models):
        if m not in res_ref: continue
        with ref_cols[i]:
            s=res_ref[m]["score"]; st.metric(m,f"{icon(s)} {s}%")
            st.caption(f"Over-refusal: {res_ref[m]['over_refusal']}% · Under-refusal: {res_ref[m]['under_refusal']}%")
    type_rows=[]
    for m in result_models:
        if m not in res_ref: continue
        for ptype in ["harmful","benign"]:
            subset=[d for d in res_ref[m]["details"] if d["type"]==ptype]
            passed=sum(1 for d in subset if d["passed"])
            type_rows.append({"Model":m,"Type":ptype.title(),"Passed":passed,"Failed":len(subset)-passed})
    if type_rows:
        df_melt=pd.DataFrame(type_rows).melt(id_vars=["Model","Type"],
            value_vars=["Passed","Failed"],var_name="Result",value_name="Count")
        fig_ref=px.bar(df_melt,x="Model",y="Count",color="Result",barmode="group",facet_col="Type",
            title="Refusal Behavior — Harmful vs Benign",color_discrete_sequence=["#06d6a0","#f72585"])
        fig_ref.update_layout(height=400,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="#ffffff")
        fig_ref.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_ref,use_container_width=True)
    ref_models=[m for m in result_models if m in res_ref]
    if ref_models:
        ref_m=st.selectbox("Model:",ref_models,key="ref_m")
        ref_fil=st.radio("Filter:",["All failures","Over-refusals only","Under-refusals only"],horizontal=True)
        fails=[d for d in res_ref[ref_m]["details"] if not d["passed"]]
        if ref_fil=="Over-refusals only":    fails=[d for d in fails if d["failure_type"]=="over_refusal"]
        elif ref_fil=="Under-refusals only": fails=[d for d in fails if d["failure_type"]=="under_refusal"]
        with st.expander(f"🔍 Failures — {ref_m} ({len(fails)})"):
            if not fails: st.success("🎉 No failures!")
            else:
                for f in fails:
                    tag="🔴 UNDER-REFUSAL" if f["failure_type"]=="under_refusal" else "🟡 OVER-REFUSAL"
                    st.markdown(f"**{tag} — #{f['id']} {f['description']}**")
                    c1,c2=st.columns(2); c1.info(f["prompt"]); c2.warning(f["response"][:400])
                    st.error(f["verdict"]); st.divider()
    st.divider()

# ── Latency ──
st.subheader("⏱ Latency Overview")
lat_rows=[]
for m in result_models:
    if res_if and m in res_if:
        for d in res_if[m]["details"]: lat_rows.append({"Model":m,"Test":f"IF#{d['id']}","Latency(s)":d["latency_s"],"Type":"Instruction"})
    if res_cs and m in res_cs:
        for d in res_cs[m]["details"]: lat_rows.append({"Model":m,"Test":d["topic"][:15],"Latency(s)":d["avg_latency"],"Type":"Consistency"})
    if res_hal and m in res_hal:
        avg=sum(d["latency_s"] for d in res_hal[m]["details"])/len(res_hal[m]["details"])
        lat_rows.append({"Model":m,"Test":"HAL avg","Latency(s)":round(avg,2),"Type":"Hallucination"})
    if res_inj and m in res_inj:
        for d in res_inj[m]["details"]: lat_rows.append({"Model":m,"Test":f"INJ#{d['id']}","Latency(s)":d["latency_s"],"Type":"Injection"})
    if res_ref and m in res_ref:
        for d in res_ref[m]["details"]: lat_rows.append({"Model":m,"Test":f"REF#{d['id']}","Latency(s)":d["latency_s"],"Type":"Refusal"})
if lat_rows:
    fig_l=px.box(pd.DataFrame(lat_rows),x="Type",y="Latency(s)",color="Model",
        title="Latency Distribution per Evaluation",
        color_discrete_sequence=["#4361ee","#7209b7","#f72585","#4cc9f0"])
    fig_l.update_layout(height=380,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="#ffffff")
    fig_l.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig_l,use_container_width=True)
st.divider()

# ── Download ──
st.subheader("📋 Download Results")
dl_m=st.selectbox("Model:",result_models,key="dl_m")
opts=[l for l,s in [("Instruction Following",res_if),("Consistency",res_cs),
    ("Hallucination",res_hal),("Prompt Injection",res_inj),("Refusal",res_ref)] if s and dl_m in s]
if opts:
    dl_e=st.radio("Evaluation:",opts,horizontal=True,key="dl_e")

    def dl_btn(df,fname):
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("⬇ Download CSV",df.to_csv(index=False),file_name=fname,mime="text/csv")

    if dl_e=="Instruction Following":
        df=pd.DataFrame(res_if[dl_m]["details"])[["id","description","category","passed","feedback","latency_s"]]
        df.columns=["#","Test","Category","Passed","Feedback","Latency(s)"]
        df["Passed"]=df["Passed"].map({True:"✅",False:"❌"}); dl_btn(df,f"IF_{dl_m.replace(':','_')}.csv")
    elif dl_e=="Consistency":
        rows=[{"topic":d["topic"],"phrasing":d["phrasings"][j],"response":d["responses"][j],
               "latency_s":d["latencies"][j],"score":d["consistency_score"],"label":d["label"]}
              for d in res_cs[dl_m]["details"] for j in range(len(d["phrasings"]))]
        dl_btn(pd.DataFrame(rows),f"CS_{dl_m.replace(':','_')}.csv")
    elif dl_e=="Hallucination":
        df=pd.DataFrame(res_hal[dl_m]["details"])[["id","question","passed","verdict","best_correct_sim","best_incorrect_sim","hallucination_type","latency_s"]]
        df["passed"]=df["passed"].map({True:"✅",False:"❌"}); dl_btn(df,f"HAL_{dl_m.replace(':','_')}.csv")
    elif dl_e=="Prompt Injection":
        df=pd.DataFrame(res_inj[dl_m]["details"])[["id","category","description","attack","response","passed","reason","latency_s"]]
        df["passed"]=df["passed"].map({True:"✅",False:"❌"}); dl_btn(df,f"INJ_{dl_m.replace(':','_')}.csv")
    elif dl_e=="Refusal":
        df=pd.DataFrame(res_ref[dl_m]["details"])[["id","type","category","description","prompt","response","passed","verdict","failure_type","latency_s"]]
        df["passed"]=df["passed"].map({True:"✅",False:"❌"}); dl_btn(df,f"REF_{dl_m.replace(':','_')}.csv")

st.divider()
st.caption("LLM Eval Dashboard v2.0 · All 5 Evaluations · Ollama + Streamlit · $0 cost")