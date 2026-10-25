"""FactVerse UI for the local and Hugging Face fake-news classifier."""

from pathlib import Path
import re
import sys

import gradio as gr
import joblib

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT))
    from src.text_clean import clean_text
else:
    _DATELINE = re.compile(r"^\s*[A-Za-z .,'/\-]{0,60}?\(reuters\)\s*[-–—]\s*", re.I)
    _REUTERS = re.compile(r"\(?\breuters\b\)?", re.I)
    _URL = re.compile(r"https?://\S+|www\.\S+", re.I)
    _EMAIL = re.compile(r"\S+@\S+\.\S+")
    _NON_ALPHA = re.compile(r"[^a-z']+")

    def clean_text(value):
        value = (value or "").lower()
        value = _DATELINE.sub(" ", value)
        value = _REUTERS.sub(" ", value)
        value = _URL.sub(" ", value)
        value = _EMAIL.sub(" ", value)
        return re.sub(r"\s+", " ", _NON_ALPHA.sub(" ", value)).strip()

MODEL_PATH = Path(__file__).with_name("fake_news_model.joblib")
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"
pipeline = joblib.load(MODEL_PATH)


def classify(title, statement):
    raw = f"{(title or '').strip()} {(statement or '').strip()}".strip()
    if not raw:
        return (
            "<div class='placeholder-panel error-panel'><div class='placeholder-icon'>!</div>"
            "<h3>Add a headline or statement</h3><p>Enter some news text to get a model prediction.</p></div>"
        )

    cleaned = clean_text(raw)
    prediction = int(pipeline.predict([cleaned])[0])
    is_fake = prediction == 1
    kind = "fake" if is_fake else "true"
    label = "FAKE" if is_fake else "TRUE"
    return (
        f"<section class='result-panel prediction-only {kind}'><div class='prediction-value'>{label}</div></section>"
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root { --navy:#050B14; --navy-light:#0D1626; --electric:#00D4FF; --amber:#FFB703; }
body,.gradio-container { background:var(--navy)!important; color:#e2e8f0!important; font-family:Inter,sans-serif!important; }
body { overflow-x:hidden; }
.gradio-container { max-width:1160px!important; margin:auto!important; padding:26px 22px 48px!important; }
.gradio-container footer { display:none!important; }
.ambient { position:fixed; inset:0; overflow:hidden; z-index:0; pointer-events:none; background:radial-gradient(ellipse at 50% 25%,#0a1a2b 0,transparent 52%); }
.ambient-orb { position:absolute; width:290px; height:290px; border-radius:50%; left:calc(50% - 145px); top:115px; opacity:.18; background:radial-gradient(circle at 32% 28%,#a6f1ff 0,#00d4ff 9%,#104f89 36%,#081a35 68%,transparent 72%); box-shadow:0 0 100px #00d4ff66; animation:orb-float 8s ease-in-out infinite; }
.ambient-ring { position:absolute; width:420px; height:160px; left:calc(50% - 210px); top:192px; border:1px solid #00d4ff33; border-radius:50%; transform:rotate(-18deg); animation:ring-spin 24s linear infinite; }
.ambient-ring:after { content:''; position:absolute; inset:-22px 28px; border:1px solid #ffb70328; border-radius:50%; transform:rotate(42deg); }
.ambient-card { position:absolute; width:100px; height:130px; top:180px; border-radius:10px; border:1px solid #5aa8c755; background:linear-gradient(145deg,#13253ab0,#08132188); box-shadow:0 16px 40px #0008; transform:perspective(700px) rotateY(-24deg) rotateZ(-8deg); animation:card-drift 9s ease-in-out infinite; }
.ambient-card:before,.ambient-card:after { content:''; position:absolute; left:13px; right:13px; height:4px; border-radius:4px; top:36px; background:#00d4ff66; box-shadow:0 14px #8aa8ff38,0 28px #8aa8ff38,0 42px #8aa8ff38; }
.ambient-card:after { width:32px; right:auto; top:18px; height:6px; background:#ffb703aa; box-shadow:none; }
.ambient-card.left { left:8%; }.ambient-card.right { right:8%; top:230px; transform:perspective(700px) rotateY(24deg) rotateZ(8deg); animation-delay:-4s; }
@keyframes orb-float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-14px)} }
@keyframes ring-spin { from{transform:rotate(-18deg)} to{transform:rotate(342deg)} }
@keyframes card-drift { 0%,100%{translate:0 0} 50%{translate:0 -16px} }
.site-header { display:flex; align-items:center; justify-content:space-between; margin:0 0 56px; position:relative; z-index:1; }
.brand { display:flex; align-items:center; gap:10px; color:#fff; font-size:21px; font-weight:700; letter-spacing:-.03em; }
.brand svg { width:29px; height:29px; color:var(--electric); }
.top-pill { color:#94a3b8; font-size:13px; font-weight:500; background:#0d1626bb; padding:10px 16px; border-radius:99px; border:1px solid #ffffff12; }
.hero { max-width:800px; margin:0 auto 42px; text-align:center; position:relative; z-index:1; }
.hero h1 { color:#fff; font-size:clamp(38px,6vw,62px); font-weight:700; line-height:1.12; letter-spacing:-.045em; margin:0 0 18px; }
.hero h1 span { color:var(--electric)!important; -webkit-text-fill-color:var(--electric)!important; background:none!important; }
.hero p { color:#94a3b8; font-size:17px; line-height:1.75; max-width:740px; margin:auto; }
.workspace { display:flex!important; gap:24px!important; align-items:stretch!important; margin:0 0 28px!important; position:relative; z-index:1; }
.workspace > .input-col { flex:7 1 0!important; min-width:0!important; }.workspace > .result-col { flex:5 1 0!important; min-width:0!important; }
.glass-panel { height:100%; padding:28px!important; border:1px solid #ffffff1c!important; border-radius:18px!important; background:rgba(13,22,38,.78)!important; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); box-shadow:0 24px 70px #0008,inset 0 1px #ffffff0a!important; }
.form-title { display:flex; align-items:center; gap:10px; margin:0 0 22px!important; color:#fff!important; font-size:20px!important; font-weight:600!important; }.form-icon { color:var(--electric); font-size:22px; }
label { color:#cbd5e1!important; font-size:14px!important; font-weight:500!important; margin-bottom:7px!important; }
input,textarea { background:#050b14bb!important; border:1px solid #ffffff1c!important; border-radius:12px!important; color:#f8fafc!important; padding:13px 15px!important; box-shadow:none!important; }
input::placeholder,textarea::placeholder { color:#64748b!important; } input:focus,textarea:focus { border-color:var(--electric)!important; box-shadow:0 0 0 2px #00d4ff26!important; }
#headline { min-height:0!important; height:auto!important; }
#headline textarea { min-height:48px!important; height:48px!important; }
#article,#article textarea { min-height:190px!important; }
.analyze-btn,.analyze-btn button { width:100%!important; min-height:54px!important; background:var(--electric)!important; color:var(--navy)!important; border:0!important; border-radius:12px!important; font-weight:700!important; font-size:16px!important; box-shadow:0 8px 30px #00d4ff27!important; transition:transform .2s,background .2s!important; }
.analyze-btn:hover,.analyze-btn button:hover { background:#fff!important; transform:translateY(-1px); }
.placeholder-panel,.result-panel { min-height:360px; height:100%; border-radius:18px; padding:28px; background:rgba(13,22,38,.78); backdrop-filter:blur(14px); border:1px solid #ffffff1c; box-shadow:0 20px 60px #0007; }
.prediction-only { display:grid; place-items:center; }
.prediction-value { font-size:clamp(48px,7vw,76px); font-weight:700; letter-spacing:.08em; text-shadow:0 0 32px currentColor; }
.prediction-only.fake .prediction-value { color:var(--amber); }
.prediction-only.true .prediction-value { color:var(--electric); }
.placeholder-panel { border-style:dashed; border-color:#ffffff30; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }
.placeholder-icon { width:62px; height:62px; display:grid; place-items:center; border-radius:50%; background:#050b14; border:1px solid #ffffff18; color:#64748b; font-size:25px; margin-bottom:16px; }
.placeholder-panel h3 { color:#cbd5e1; font-size:18px; font-weight:500; margin:0 0 7px; }.placeholder-panel p { color:#64748b; font-size:14px; margin:0; }
.error-panel { border-color:#ffb70370; }.error-panel .placeholder-icon { color:var(--amber); }
.result-panel.fake { border-color:#ffb70355; box-shadow:0 20px 60px #ffb7030c; }.result-panel.true { border-color:#00d4ff55; box-shadow:0 20px 60px #00d4ff0c; }
.result-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:28px; }
.eyebrow,.insight>span { display:block; color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:.13em; font-weight:600; margin-bottom:7px; }
.result-heading h3 { color:#f8fafc; font-size:20px; line-height:1.35; font-weight:700; margin:0; }
.result-badge { flex:none; padding:6px 11px; border-radius:99px; background:#ffb70320; color:var(--amber); border:1px solid #ffb70375; font-size:13px; font-weight:600; }.true .result-badge { color:var(--electric); background:#00d4ff19; border-color:#00d4ff77; }
.score-row { display:flex; justify-content:space-between; align-items:center; margin:0 0 9px; color:#cbd5e1; font-size:14px; }.score-row small { color:#64748b; }.score-row b { color:#f8fafc; font-size:16px; }
.score-track { height:10px; overflow:hidden; border-radius:99px; background:#050b14; border:1px solid #ffffff0d; margin-bottom:26px; }.score-track i { display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#ffb70399,#ffb703); transition:width .8s ease; }.true .score-track i { background:linear-gradient(90deg,#00d4ff99,#00d4ff); }
.insight { background:#050b1480; border:1px solid #ffffff0c; border-radius:12px; padding:16px; }.insight p { color:#cbd5e1; font-size:14px; line-height:1.65; margin:0; }
.dataset-title { margin:0 0 14px; color:#94a3b8; font-size:12px; font-weight:600; letter-spacing:.13em; text-transform:uppercase; }
.dataset-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:28px; position:relative; z-index:1; }
.dataset-card { display:flex; align-items:center; gap:15px; padding:18px 20px; border:1px solid #ffffff1c; border-radius:14px; background:rgba(13,22,38,.76); backdrop-filter:blur(12px); }
.dataset-icon { width:46px; height:46px; flex:none; display:grid; place-items:center; border-radius:12px; color:var(--electric); background:#00d4ff12; border:1px solid #00d4ff35; font-size:22px; }.dataset-card.fake .dataset-icon { color:var(--amber); background:#ffb70312; border-color:#ffb70335; }
.dataset-card b { display:block; color:#fff; font-size:22px; line-height:1.15; }.dataset-card span { color:#94a3b8; font-size:13px; }
.legal-note { display:flex; align-items:center; justify-content:center; gap:9px; text-align:center; color:#94a3b8; font-size:13px; line-height:1.6; padding:17px 20px; border:1px solid #ffffff12; border-radius:13px; background:rgba(13,22,38,.58); position:relative; z-index:1; }.legal-note .info { color:#64748b; font-size:17px; }
.gradio-container .form { gap:15px!important; }
@media(max-width:820px){ .gradio-container{padding:20px 16px 36px!important}.site-header{margin-bottom:42px}.hero{margin-bottom:34px}.workspace{flex-direction:column!important}.workspace>.input-col,.workspace>.result-col{width:100%!important;flex:none!important}.placeholder-panel,.result-panel{min-height:260px}.ambient-card.left{left:-55px}.ambient-card.right{right:-55px} }
@media(max-width:560px){ .gradio-container{padding:16px 14px 28px!important}.site-header{margin-bottom:38px}.brand{font-size:19px}.top-pill{font-size:11px;padding:8px 11px}.hero h1{font-size:39px}.hero p{font-size:15px}.glass-panel{padding:20px!important}.dataset-grid{grid-template-columns:1fr;gap:10px}.result-heading{flex-direction:column}.legal-note{align-items:flex-start;text-align:left}.ambient-orb{top:130px;width:220px;height:220px;left:calc(50% - 110px)}.ambient-ring{top:170px;width:310px;left:calc(50% - 155px)} }
@media(prefers-reduced-motion:reduce){ *,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important} }
"""

PLACEHOLDER = (
    "<div class='placeholder-panel'><div class='placeholder-icon'>ⓘ</div>"
    "<h3>Awaiting Analysis</h3><p>Your fact-check results will appear here.</p></div>"
)

with gr.Blocks(css=CSS, title="FactVerse · AI Fake News Detection", theme=gr.themes.Base()) as demo:
    gr.HTML(
        "<div class='ambient'><div class='ambient-orb'></div><div class='ambient-ring'></div>"
        "<div class='ambient-card left'></div><div class='ambient-card right'></div></div>"
        "<header class='site-header'><div class='brand'><svg fill='none' stroke='currentColor' viewBox='0 0 24 24'><path stroke-width='2' stroke-linecap='round' stroke-linejoin='round' d='M9 12l2 2 4-4m5.6-4A12 12 0 0112 3a12 12 0 01-8.6 3A12 12 0 003 9c0 5.6 3.8 10.3 9 11.6 5.2-1.3 9-6 9-11.6 0-1-.1-2-.4-3z'/></svg>FactVerse</div>"
        "<div class='top-pill'>AI Fact-Checking Studio</div></header>"
        "<section class='hero'><h1>Check the story.<br><span>See what the model finds.</span></h1>"
        "<p>Paste a headline and article to uncover language patterns our machine-learning model recognizes in its training data.</p></section>"
    )
    with gr.Row(elem_classes="workspace"):
        with gr.Column(scale=7, min_width=320, elem_classes="input-col"):
            with gr.Column(elem_classes="glass-panel"):
                gr.HTML(
                    "<h2 class='form-title'><span class='form-icon'>✎</span>Submit Article</h2>"
                )
                title = gr.Textbox(label="Headline", placeholder="e.g. Breaking: Local report reveals…", lines=1, elem_id="headline")
                statement = gr.Textbox(label="Article Text", placeholder="Paste the news statement or full article here…", lines=8, elem_id="article")
                check = gr.Button("Check News", variant="primary", elem_classes="analyze-btn")
        with gr.Column(scale=5, min_width=300, elem_classes="result-col"):
            output = gr.HTML(value=PLACEHOLDER)
    check.click(classify, inputs=[title, statement], outputs=output)
    statement.submit(classify, inputs=[title, statement], outputs=output)
    gr.HTML(
        "<div style='position:relative;z-index:1'><p class='dataset-title'>Training Dataset</p>"
        "<section class='dataset-grid'><div class='dataset-card'><div class='dataset-icon'>✓</div><div><b>21,417</b><span>True articles</span></div></div>"
        "<div class='dataset-card fake'><div class='dataset-icon'>!</div><div><b>23,481</b><span>Fake articles</span></div></div></section>"
        "<div class='legal-note'><span class='info'>ⓘ</span>This demo learns patterns from its training data. It can be wrong and does not independently verify claims.</div></div>"
    )


if __name__ == "__main__":
    demo.launch()
