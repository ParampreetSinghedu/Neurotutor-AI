import os
import json
import base64
import random
import datetime
import io
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="NeuroTutor AI", page_icon="🧠", layout="wide")

# ── Gemini Client ─────────────────────────────────────────────────────────────
client = genai.Client(
    api_key=os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY"),
    http_options={
        "api_version": "",
        "base_url": os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL"),
    },
)


def gemini(prompt: str) -> str:
    try:
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=8192),
        )
        return r.text or ""
    except Exception as e:
        if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
            return "⚠️ Your cloud budget has been exceeded. Please check your Replit credits."
        return f"⚠️ Error: {e}"


# ── Login History (JSON file) ─────────────────────────────────────────────────
LOGIN_HISTORY_FILE = "login_history.json"


def load_login_history() -> list:
    if os.path.exists(LOGIN_HISTORY_FILE):
        try:
            with open(LOGIN_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_login_entry(name: str, email: str):
    history = load_login_history()
    history.append({
        "name": name,
        "email": email,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    history = history[-50:]  # keep last 50
    with open(LOGIN_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ── File Text Extraction ──────────────────────────────────────────────────────
def extract_pdf_text(file) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        return f"Error reading PDF: {e}"


def extract_docx_text(file) -> str:
    try:
        from docx import Document
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as e:
        return f"Error reading Word file: {e}"


# ── Voice Transcription ───────────────────────────────────────────────────────
def transcribe_audio(audio_file) -> tuple[str, str]:
    """Returns (text, error). Uses SpeechRecognition with Google Web Speech."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text, ""
    except Exception as e:
        err = str(e)
        if "UnknownValueError" in err or "could not understand" in err.lower():
            return "", "Could not understand the audio. Please speak clearly and try again."
        if "RequestError" in err:
            return "", "Speech recognition service unavailable. Please check your internet connection."
        return "", f"Transcription error: {err}"


# ── Session State Init ────────────────────────────────────────────────────────
def init():
    defaults = {
        "screen": "welcome",
        "user": {},
        "subjects": [],
        "notes": {},
        "analysis_results": [],
        "calendar_events": [],
        "mood_history": [],
        "companion_chat": [],
        "emotional_chat": [],
        "dashboard_section": "🏠 Home",
        "exam_questions": [],
        "exam_topic": "",
        "reexplain_topic": "",
        "reexplain_original": "",
        "game_qs": [],
        "game_idx": 0,
        "game_score": 0,
        "game_done": False,
        "game_answers": [],
        "today_mood": False,
        "num_subjects": 0,
        "last_file_text": "",       # text extracted from last uploaded file
        "last_file_name": "",       # filename of last upload
        "voice_transcript": "",     # latest voice transcript
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


# ── Helpers ───────────────────────────────────────────────────────────────────
def nav(screen):
    st.session_state.screen = screen
    st.rerun()


def exam_mode_active():
    today = datetime.date.today()
    for ev in st.session_state.calendar_events:
        if ev["type"] == "Exam":
            try:
                d = datetime.date.fromisoformat(ev["date"])
                if 0 <= (d - today).days <= 5:
                    return ev["title"]
            except Exception:
                pass
    return None


# ── WELCOME ───────────────────────────────────────────────────────────────────
def show_welcome():
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🧠 NeuroTutor AI")
        st.markdown("### *Students teach. AI learns. Gaps disappear.*")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "> *NeuroTutor AI is not just a learning app — it is a self-awareness engine "
            "that reveals what students truly understand and guides their academic and mental growth.*"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            nav("register")


# ── REGISTER ─────────────────────────────────────────────────────────────────
def show_register():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 📋 Create Your Profile")
        st.markdown("Tell us about yourself so we can personalize your experience.")
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("reg_form"):
            name = st.text_input("👤 Full Name", placeholder="Your name")
            email = st.text_input("📧 Email", placeholder="your@email.com")
            age = st.number_input("🎂 Age", min_value=5, max_value=100, value=16)
            gender = st.selectbox("⚧ Gender", ["Prefer not to say", "Male", "Female", "Other"])
            num_subjects = st.number_input("📚 How many subjects do you study?", min_value=1, max_value=15, value=3)
            submitted = st.form_submit_button("Continue →", type="primary", use_container_width=True)

        if submitted:
            if not name.strip() or not email.strip():
                st.warning("Please fill in your name and email.")
            else:
                st.session_state.user = {
                    "name": name.strip(), "email": email.strip(),
                    "age": int(age), "gender": gender,
                }
                st.session_state.num_subjects = int(num_subjects)
                save_login_entry(name.strip(), email.strip())
                nav("subjects")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            nav("welcome")

        # ── Previous Users ──
        history = load_login_history()
        if history:
            st.markdown("---")
            st.markdown("### 🕓 Previous Users")
            for entry in reversed(history[-5:]):
                st.markdown(f"👤 **{entry['name']}** ({entry['email']}) — {entry['time']}")


# ── SUBJECT SETUP ─────────────────────────────────────────────────────────────
def show_subject_setup():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        n = st.session_state.num_subjects
        st.markdown(f"## 📚 Set Up Your {n} Subject{'s' if n > 1 else ''}")
        st.markdown("Name each subject you want to track.")
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("subj_form"):
            subjects = [
                st.text_input(f"Subject {i+1}", placeholder="e.g. Mathematics, Physics…", key=f"subj_{i}")
                for i in range(n)
            ]
            submitted = st.form_submit_button("Enter Dashboard →", type="primary", use_container_width=True)

        if submitted:
            filled = [s.strip() for s in subjects if s.strip()]
            if len(filled) < n:
                st.warning(f"Please name all {n} subjects.")
            else:
                st.session_state.subjects = filled
                st.session_state.notes = {s: [] for s in filled}
                st.session_state.dashboard_section = "🏠 Home"
                nav("dashboard")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            nav("register")


# ── DASHBOARD SIDEBAR ─────────────────────────────────────────────────────────
SECTIONS = ["🏠 Home", "🧠 AI Teacher", "📂 Notes", "📅 Calendar", "💬 AI Companion", "🎮 Mini Game", "📊 Mood Tracker"]


def dashboard_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"### 👤 {user.get('name', 'Student')}")
        st.markdown(f"*{user.get('email', '')}*")
        st.markdown("---")
        for section in SECTIONS:
            active = st.session_state.dashboard_section == section
            if st.button(section, use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.dashboard_section = section
                st.rerun()
        st.markdown("---")
        exam = exam_mode_active()
        if exam:
            st.error(f"🚨 EXAM MODE\n\n**{exam}** is within 5 days!")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init()
            nav("welcome")


# ── HOME ──────────────────────────────────────────────────────────────────────
def show_home():
    user = st.session_state.user
    st.markdown(f"# 🏠 Welcome back, {user['name']}! 👋")
    st.markdown("*Students teach. AI learns. Gaps disappear.*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Subjects", len(st.session_state.subjects))
    c2.metric("🧠 Analyses Done", len(st.session_state.analysis_results))
    c3.metric("📅 Events", len(st.session_state.calendar_events))
    c4.metric("📊 Mood Logs", len(st.session_state.mood_history))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📚 Your Subjects & Gap Status")
        if st.session_state.subjects:
            for s in st.session_state.subjects:
                results = [r for r in st.session_state.analysis_results if r.get("subject") == s]
                gap = results[-1]["gap"] if results else "🔘 Not analyzed"
                st.markdown(f"**{s}** — {gap}")
        else:
            st.info("No subjects set up yet.")

    with col2:
        st.markdown("### 📅 Upcoming Events")
        today = datetime.date.today()
        upcoming = sorted(
            [e for e in st.session_state.calendar_events if datetime.date.fromisoformat(e["date"]) >= today],
            key=lambda x: x["date"],
        )[:5]
        if upcoming:
            for e in upcoming:
                icon = {"Exam": "🎯", "Assignment": "📝", "Event": "📌"}.get(e["type"], "📌")
                st.markdown(f"{icon} **{e['title']}** — {e['date']}")
        else:
            st.info("No upcoming events.")

    exam = exam_mode_active()
    if exam:
        st.markdown("---")
        st.error(f"🚨 **EXAM MODE ACTIVATED** — *{exam}* is in 5 days or less!")
        st.markdown("Focus on practice questions and revision. Head to **AI Teacher → Examiner** to start!")


# ── AI TEACHER ────────────────────────────────────────────────────────────────
def show_ai_teacher():
    st.markdown("# 🧠 AI Teacher Mode")
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Analyze", "🔄 Re-Explain", "📝 Examiner", "📊 Gap Status"])

    # ── Analyze ──────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Teach the AI — It Finds Your Gaps")
        subject = st.selectbox("Subject", st.session_state.subjects or ["General"], key="analyze_subject")
        topic = st.text_input("Topic", placeholder="e.g. Newton's Laws of Motion")

        # Voice input
        st.markdown("##### 🎤 Voice Input *(optional)*")
        st.caption("Record yourself explaining the concept — it will be transcribed automatically.")
        audio_file = st.audio_input("Click to record", key="voice_input")

        if audio_file is not None:
            if st.button("📝 Transcribe Recording", key="transcribe_btn"):
                with st.spinner("Transcribing your voice…"):
                    text, err = transcribe_audio(audio_file)
                if err:
                    st.warning(f"⚠️ {err}")
                elif text:
                    st.session_state.voice_transcript = text
                    st.success(f"✅ Transcribed: *{text[:120]}{'…' if len(text) > 120 else ''}*")

        # Text explanation — pre-fill from voice if available
        default_text = st.session_state.voice_transcript
        explanation = st.text_area(
            "Your explanation (type or use voice above):",
            value=default_text,
            height=180,
            placeholder="Write or paste your explanation here…",
            key="explain_area",
        )

        # Show context from uploaded notes file
        file_context = ""
        if st.session_state.last_file_text:
            with st.expander(f"📎 Using uploaded file: *{st.session_state.last_file_name}*"):
                st.text(st.session_state.last_file_text[:800] + ("…" if len(st.session_state.last_file_text) > 800 else ""))
            file_context = f"\n\nAdditionally, the student has uploaded notes:\n\"\"\"{st.session_state.last_file_text[:3000]}\"\"\""

        if st.button("🔬 Analyze My Explanation", type="primary"):
            if not explanation.strip():
                st.warning("Please write or record your explanation first.")
            else:
                with st.spinner("Analyzing…"):
                    prompt = f"""A student is explaining the topic: "{topic or 'general'}" under subject "{subject}".
Their spoken/written explanation: \"\"\"{explanation}\"\"\"{file_context}

Provide a structured analysis with exactly these sections:
**1. Missing Concepts** (bullet points — what important ideas are absent)
**2. Incorrect Logic** (bullet points — factual or reasoning errors; say "None found ✅" if none)
**3. Suggestions to Improve** (bullet points — clear, actionable advice)
**4. Confidence Score** — give a number 0–100 then one label: 🟢 Strong (80–100), 🟡 Medium (40–79), 🔴 Weak (0–39)

Be encouraging, clear, and concise."""
                    result = gemini(prompt)

                gap = "🟢 Strong"
                if "🔴 Weak" in result:
                    gap = "🔴 Weak"
                elif "🟡 Medium" in result:
                    gap = "🟡 Medium"

                st.session_state.analysis_results.append({
                    "subject": subject, "topic": topic, "explanation": explanation,
                    "result": result, "gap": gap, "date": str(datetime.date.today()),
                })
                st.session_state.reexplain_topic = topic
                st.session_state.reexplain_original = explanation
                st.session_state.voice_transcript = ""

                st.markdown("---")
                st.markdown("#### 📋 Analysis Results")
                st.markdown(result)

    # ── Re-Explain ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Re-Explain Mode")
        st.markdown("Try explaining the same concept in a simpler, clearer way.")
        topic = st.text_input("Topic to re-explain", value=st.session_state.reexplain_topic, key="re_topic")
        original = st.text_area("Your original explanation:", value=st.session_state.reexplain_original, height=100)
        new_explanation = st.text_area("Explain it again — simpler:", height=160, placeholder="Use simpler words, analogies, examples…")
        if st.button("🔄 Evaluate Re-Explanation", type="primary"):
            if not new_explanation.strip():
                st.warning("Please write your new explanation.")
            else:
                with st.spinner("Evaluating…"):
                    prompt = f"""A student is re-explaining the topic: "{topic}".
Original: \"\"\"{original}\"\"\"
New simpler version: \"\"\"{new_explanation}\"\"\"

Compare the two. Is the new explanation clearer? What improved? What still needs work?
Give brief, encouraging feedback with bullet points."""
                    result = gemini(prompt)
                st.markdown("---")
                st.markdown(result)

    # ── Examiner ──────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📝 Examiner Mode")
        st.markdown("The AI will ask you questions to test your understanding.")
        e_topic = st.text_input("Enter a topic to be examined on:", key="exam_topic_input")
        e_subject = st.selectbox("Subject", st.session_state.subjects or ["General"], key="exam_subject")
        if st.button("Generate Questions", type="primary"):
            if not e_topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating questions…"):
                    prompt = f"""Generate exactly 3 short-answer exam questions about "{e_topic}" (subject: {e_subject}).
Number them 1, 2, 3. Keep each question on one line. No answers — just the questions."""
                    qs = gemini(prompt)
                    st.session_state.exam_questions = [
                        q.strip() for q in qs.strip().split("\n")
                        if q.strip() and q.strip()[0].isdigit()
                    ]
                    st.session_state.exam_topic = e_topic

        if st.session_state.exam_questions:
            st.markdown("---")
            st.markdown(f"**Topic: {st.session_state.exam_topic}**")
            answers = []
            for i, q in enumerate(st.session_state.exam_questions):
                st.markdown(f"**{q}**")
                ans = st.text_area("Your answer:", key=f"exam_ans_{i}", height=80)
                answers.append(ans)

            if st.button("✅ Submit Answers", type="primary"):
                with st.spinner("Evaluating your answers…"):
                    qa = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {answers[i]}" for i, q in enumerate(st.session_state.exam_questions)])
                    prompt = f"""A student answered these exam questions about "{st.session_state.exam_topic}":
{qa}

For each answer: what's correct, what's missing, score out of 10.
End with an overall score out of {len(st.session_state.exam_questions) * 10}. Be encouraging."""
                    result = gemini(prompt)
                st.markdown("---")
                st.markdown("#### Examiner Feedback")
                st.markdown(result)

    # ── Gap Status ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📊 Knowledge Gap Status")
        if not st.session_state.analysis_results:
            st.info("No analyses yet. Go to the **Analyze** tab to get started!")
        else:
            for subject in st.session_state.subjects:
                results = [r for r in st.session_state.analysis_results if r.get("subject") == subject]
                if results:
                    latest = results[-1]
                    st.markdown(f"**{subject}** — {latest['gap']}")
                    with st.expander(f"Last analysis: {latest['topic']} ({latest['date']})"):
                        st.markdown(latest["result"])
            st.markdown("---")
            st.markdown("**Legend:** 🟢 Strong (80–100) | 🟡 Medium (40–79) | 🔴 Weak (0–39)")


# ── NOTES ─────────────────────────────────────────────────────────────────────
def show_notes():
    st.markdown("# 📂 Notes System")
    if not st.session_state.subjects:
        st.info("Please set up subjects first.")
        return

    subject = st.selectbox("📚 Select Subject", st.session_state.subjects)
    notes_list = st.session_state.notes.get(subject, [])

    tab_write, tab_upload = st.tabs(["✏️ Write Notes", "📁 Upload File"])

    # ── Write Notes ───────────────────────────────────────────────────────────
    with tab_write:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### Add Note — *{subject}*")
            note_title = st.text_input("Note Title", placeholder="e.g. Chapter 3 Summary")
            note_content = st.text_area("Note Content", height=200, placeholder="Write your notes here…")
            if st.button("💾 Save Note", type="primary"):
                if note_title.strip() and note_content.strip():
                    st.session_state.notes[subject].append(
                        {"title": note_title.strip(), "content": note_content.strip(), "source": "written"}
                    )
                    st.success("Note saved!")
                    st.rerun()
                else:
                    st.warning("Please fill in both title and content.")

        with col2:
            st.markdown(f"### Saved Notes ({len(notes_list)})")
            if notes_list:
                for i, note in enumerate(notes_list):
                    src_icon = "📁" if note.get("source") == "file" else "✏️"
                    with st.expander(f"{src_icon} {note['title']}"):
                        st.write(note["content"][:500] + ("…" if len(note["content"]) > 500 else ""))
                        if st.button("🗑️ Delete", key=f"del_{subject}_{i}"):
                            st.session_state.notes[subject].pop(i)
                            st.rerun()
            else:
                st.info("No notes yet.")

        # ── AI Note Analysis ─────────────────────────────────────────────────
        if notes_list:
            st.markdown("---")
            st.markdown("### 🤖 Analyze a Note with AI")
            selected = st.selectbox("Choose a note", [n["title"] for n in notes_list], key="note_select")
            note_obj = next((n for n in notes_list if n["title"] == selected), None)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📊 Summarize", type="primary", key="summarize_btn"):
                    with st.spinner("Summarizing…"):
                        result = gemini(f"Summarize these student notes concisely with bullet points:\n\n{note_obj['content']}")
                    st.markdown("#### Summary")
                    st.markdown(result)
            with c2:
                if st.button("💡 Suggest Improvements", key="improve_btn"):
                    with st.spinner("Analyzing…"):
                        result = gemini(f"Review these student notes and suggest improvements, additions, and corrections:\n\n{note_obj['content']}")
                    st.markdown("#### Suggestions")
                    st.markdown(result)

    # ── File Upload ───────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown(f"### 📁 Upload File — *{subject}*")
        st.caption("Upload a PDF or Word document to extract text and analyze with AI.")

        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx"],
            help="Supported: PDF (.pdf) and Word (.docx)",
        )

        if uploaded:
            fname = uploaded.name
            ext = fname.rsplit(".", 1)[-1].lower()

            with st.spinner(f"Extracting text from *{fname}*…"):
                if ext == "pdf":
                    text = extract_pdf_text(uploaded)
                elif ext == "docx":
                    text = extract_docx_text(uploaded)
                else:
                    text = ""

            if text.startswith("Error"):
                st.error(text)
            elif not text.strip():
                st.warning("No text could be extracted from this file.")
            else:
                st.success(f"✅ Extracted {len(text)} characters from *{fname}*")
                st.session_state.last_file_text = text
                st.session_state.last_file_name = fname

                with st.expander("👁️ Preview extracted text"):
                    st.text(text[:1500] + ("…" if len(text) > 1500 else ""))

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("💾 Save as Note", type="primary", key="save_file_note"):
                        st.session_state.notes[subject].append({
                            "title": fname,
                            "content": text,
                            "source": "file",
                        })
                        st.success(f"Saved '{fname}' as a note under {subject}!")
                with col_b:
                    if st.button("📊 Summarize File", key="sum_file"):
                        with st.spinner("Summarizing…"):
                            result = gemini(
                                f"Summarize the following document content in clear bullet points. "
                                f"Highlight key concepts:\n\n{text[:4000]}"
                            )
                        st.markdown("#### Summary")
                        st.markdown(result)
                with col_c:
                    if st.button("🔍 Detect Key Concepts", key="detect_file"):
                        with st.spinner("Detecting concepts…"):
                            result = gemini(
                                f"List the key concepts, terms, and important ideas from this document. "
                                f"Group them logically:\n\n{text[:4000]}"
                            )
                        st.markdown("#### Key Concepts")
                        st.markdown(result)

                if st.button("🤖 Full AI Analysis of File", key="full_ai_file"):
                    with st.spinner("Running full analysis…"):
                        result = gemini(
                            f"Analyze these student notes/document thoroughly:\n\n{text[:4000]}\n\n"
                            f"Provide:\n1. **Summary** (3-4 bullet points)\n"
                            f"2. **Key Concepts** detected\n"
                            f"3. **Missing Topics** — what important areas are not covered?\n"
                            f"4. **Suggestions** — how to improve these notes"
                        )
                    st.markdown("#### Full Analysis")
                    st.markdown(result)


# ── CALENDAR ──────────────────────────────────────────────────────────────────
def show_calendar():
    st.markdown("# 📅 Smart Calendar")

    exam = exam_mode_active()
    if exam:
        st.error(f"🚨 **EXAM MODE** — *{exam}* is within 5 days! Focus on revision and practice.")
        with st.expander("📝 Get Practice Questions"):
            if st.button("Generate Practice Questions", type="primary", key="exam_mode_qs"):
                with st.spinner("Generating…"):
                    result = gemini(f"Generate 5 practice questions for the exam on: {exam}. Include a brief answer hint for each.")
                st.markdown(result)
        st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### ➕ Add Event")
        with st.form("cal_form"):
            event_title = st.text_input("Event Title", placeholder="e.g. Physics Final Exam")
            event_date = st.date_input("Date", min_value=datetime.date.today())
            event_type = st.selectbox("Type", ["Exam", "Assignment", "Event"])
            submitted = st.form_submit_button("Add Event", type="primary")
        if submitted and event_title.strip():
            st.session_state.calendar_events.append({
                "title": event_title.strip(), "date": str(event_date), "type": event_type,
            })
            st.success("Event added!")
            st.rerun()

    with col2:
        st.markdown("### 📋 Upcoming Events")
        today = datetime.date.today()
        events = sorted(st.session_state.calendar_events, key=lambda x: x["date"])
        if events:
            for i, e in enumerate(events):
                icon = {"Exam": "🎯", "Assignment": "📝", "Event": "📌"}.get(e["type"], "📌")
                d = datetime.date.fromisoformat(e["date"])
                days_left = (d - today).days
                tag = f"({days_left}d away)" if days_left >= 0 else "(past)"
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"{icon} **{e['title']}** — {e['date']} {tag}")
                with c2:
                    if st.button("✕", key=f"del_ev_{i}"):
                        st.session_state.calendar_events.pop(i)
                        st.rerun()
        else:
            st.info("No events yet.")

    st.markdown("---")
    st.markdown("### 🤖 AI Study Planner")
    if st.button("💡 Get AI Study Suggestions", type="primary"):
        events_text = "\n".join([f"- {e['title']} ({e['type']}) on {e['date']}" for e in st.session_state.calendar_events]) or "No events."
        subjects_text = ", ".join(st.session_state.subjects) or "No subjects."
        prompt = f"""Student's upcoming events:
{events_text}

Subjects: {subjects_text}
Today: {datetime.date.today()}

Give a smart, personalized 3-day study plan. Warn if overloaded. Suggest what to study today."""
        with st.spinner("Planning your schedule…"):
            result = gemini(prompt)
        st.markdown(result)


# ── AI COMPANION ──────────────────────────────────────────────────────────────
def show_companion():
    st.markdown("# 💬 AI Companion Mode")
    tab1, tab2 = st.tabs(["🧠 Motivation Mode", "❤️ Emotional Support"])

    with tab1:
        st.markdown("### 🧠 Motivation & Study Planning")
        st.markdown("Ask for study tips, career advice, to-do suggestions, or a motivational boost!")
        user_msg = st.text_input("Ask me anything:", key="motiv_input", placeholder="e.g. How do I stay focused?")
        if st.button("Ask", type="primary", key="motiv_btn"):
            if user_msg.strip():
                subjects = ", ".join(st.session_state.subjects) or "general subjects"
                prompt = f"""You are a supportive, motivational AI tutor and study coach.
The student studies: {subjects}. They have done {len(st.session_state.analysis_results)} analyses.
Question: "{user_msg}"
Give practical, encouraging, concise advice. Use bullet points where helpful."""
                with st.spinner("Thinking…"):
                    result = gemini(prompt)
                st.markdown("---")
                st.markdown(result)

    with tab2:
        st.markdown("### ❤️ Emotional Support")
        st.markdown("Talk to me about anything — stress, tiredness, or how your day went.")

        for msg in st.session_state.emotional_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Tell me how you're feeling…")
        if user_msg:
            st.session_state.emotional_chat.append({"role": "user", "content": user_msg})
            keywords = ["sad", "stress", "tired", "anxious", "worried", "depressed", "overwhelmed", "exhausted"]
            needs_care = any(k in user_msg.lower() for k in keywords)
            mood_context = "The student seems emotionally stressed. " if needs_care else ""
            prompt = f"""{mood_context}You are a warm, empathetic AI companion for a student.
Respond to: "{user_msg}"
If sad/stressed/tired: validate feelings, offer comfort, suggest a break or the mini game.
Keep warm, human, and concise (2-4 sentences)."""
            with st.spinner(""):
                reply = gemini(prompt)
            st.session_state.emotional_chat.append({"role": "assistant", "content": reply})
            if needs_care:
                st.session_state.emotional_chat.append({
                    "role": "assistant",
                    "content": "🎮 *Why not take a 5-minute break with the Mini Game? It's a great way to refresh your mind!*",
                })
            st.rerun()


# ── MINI GAME ─────────────────────────────────────────────────────────────────
def show_mini_game():
    st.markdown("# 🎮 Mini Quiz Game")
    st.markdown("Refresh your mind with a quick knowledge challenge!")

    if not st.session_state.game_qs:
        topic = st.text_input("Enter a topic for your quiz:", placeholder="e.g. Solar System, World History…")
        if st.button("🎯 Generate Quiz", type="primary"):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating quiz…"):
                    prompt = f"""Create a multiple choice quiz with exactly 5 questions about: "{topic}"
Format EXACTLY like this with a blank line between questions:
Q: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]"""
                    raw = gemini(prompt)
                    blocks = [b.strip() for b in raw.strip().split("\n\n") if "Q:" in b]
                    parsed = []
                    for block in blocks:
                        lines = block.split("\n")
                        try:
                            q = next(l for l in lines if l.startswith("Q:")).replace("Q:", "").strip()
                            opts = {l[0]: l[3:].strip() for l in lines if len(l) > 2 and l[1] == ")"}
                            ans = next(l for l in lines if l.startswith("Answer:")).replace("Answer:", "").strip()[0].upper()
                            if q and opts and ans:
                                parsed.append({"q": q, "opts": opts, "ans": ans})
                        except Exception:
                            pass
                    if parsed:
                        st.session_state.game_qs = parsed
                        st.session_state.game_idx = 0
                        st.session_state.game_score = 0
                        st.session_state.game_done = False
                        st.session_state.game_answers = []
                        st.rerun()
                    else:
                        st.error("Couldn't parse quiz. Please try a different topic.")
        return

    if st.session_state.game_done:
        total = len(st.session_state.game_qs)
        score = st.session_state.game_score
        pct = int(score / total * 100)
        emoji, msg = ("🏆", "Excellent!") if pct >= 80 else ("👍", "Good job!") if pct >= 50 else ("💪", "Keep practicing!")
        st.markdown(f"## {emoji} Quiz Complete! Score: **{score}/{total}** ({pct}%)")
        st.markdown(f"*{msg}*")
        st.markdown("---")
        for i, (item, chosen) in enumerate(zip(st.session_state.game_qs, st.session_state.game_answers)):
            icon = "✅" if chosen == item["ans"] else "❌"
            st.markdown(f"**Q{i+1}: {item['q']}**")
            st.markdown(f"{icon} You: **{chosen}** | Correct: **{item['ans']}** — {item['opts'].get(item['ans'], '')}")
        if st.button("🔄 Play Again", type="primary"):
            st.session_state.game_qs = []
            st.rerun()
        return

    idx = st.session_state.game_idx
    total = len(st.session_state.game_qs)
    item = st.session_state.game_qs[idx]
    st.markdown(f"**Question {idx + 1} of {total}**")
    st.progress(idx / total)
    st.markdown(f"### {item['q']}")
    choice = st.radio("Choose:", list(item["opts"].keys()), format_func=lambda k: f"{k}) {item['opts'][k]}", key=f"game_q_{idx}")
    if st.button("Submit Answer →", type="primary"):
        st.session_state.game_answers.append(choice)
        if choice == item["ans"]:
            st.session_state.game_score += 1
        if idx + 1 >= total:
            st.session_state.game_done = True
        else:
            st.session_state.game_idx += 1
        st.rerun()


# ── MOOD TRACKER ──────────────────────────────────────────────────────────────
def show_mood_tracker():
    st.markdown("# 📊 Daily Mood Tracker")
    today = str(datetime.date.today())
    already = any(m["date"] == today for m in st.session_state.mood_history)

    if not already:
        st.markdown("### How are you feeling today?")
        with st.form("mood_form"):
            mood = st.select_slider("😊 Mood", options=["😢 Very Low", "😞 Low", "😐 Okay", "🙂 Good", "😄 Great"])
            stress = st.select_slider("😤 Stress Level", options=["Very Low", "Low", "Moderate", "High", "Very High"])
            focus = st.select_slider("🎯 Focus Level", options=["Very Low", "Low", "Moderate", "High", "Very High"])
            submitted = st.form_submit_button("Log My Mood", type="primary")
        if submitted:
            st.session_state.mood_history.append({"date": today, "mood": mood, "stress": stress, "focus": focus})
            with st.spinner("Getting your suggestion…"):
                result = gemini(f"Student mood: {mood}, Stress: {stress}, Focus: {focus}. Give 3-4 bullet points: should they study, rest, or balance? One self-care tip. One study tip if mood allows.")
            st.markdown("### 💡 Today's Suggestion")
            st.markdown(result)
    else:
        entry = next(m for m in st.session_state.mood_history if m["date"] == today)
        st.success(f"✅ Today's mood logged — {entry['mood']} | Stress: {entry['stress']} | Focus: {entry['focus']}")

    if st.session_state.mood_history:
        st.markdown("---")
        st.markdown("### 📈 Mood History")
        for entry in reversed(st.session_state.mood_history[-10:]):
            st.markdown(f"**{entry['date']}** — {entry['mood']} | Stress: {entry['stress']} | Focus: {entry['focus']}")


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def show_dashboard():
    dashboard_sidebar()
    section = st.session_state.dashboard_section
    if section == "🏠 Home":
        show_home()
    elif section == "🧠 AI Teacher":
        show_ai_teacher()
    elif section == "📂 Notes":
        show_notes()
    elif section == "📅 Calendar":
        show_calendar()
    elif section == "💬 AI Companion":
        show_companion()
    elif section == "🎮 Mini Game":
        show_mini_game()
    elif section == "📊 Mood Tracker":
        show_mood_tracker()


# ── ROUTER ────────────────────────────────────────────────────────────────────
screen = st.session_state.screen
if screen == "welcome":
    show_welcome()
elif screen == "register":
    show_register()
elif screen == "subjects":
    show_subject_setup()
elif screen == "dashboard":
    show_dashboard()
