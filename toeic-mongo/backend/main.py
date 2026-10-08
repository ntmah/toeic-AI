"""
main.py – FastAPI + MongoDB Atlas + LangChain + LangSmith
Thay thế toàn bộ Supabase bằng MongoDB Motor (async).
"""

import os, uuid
from datetime import datetime, timezone
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing ─────────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))
os.environ.setdefault("LANGCHAIN_PROJECT",    os.getenv("LANGCHAIN_PROJECT", "toeic-ai-agent"))
os.environ.setdefault("LANGCHAIN_API_KEY",    os.getenv("LANGCHAIN_API_KEY", ""))

from database import create_indexes, users_col, stats_col, sessions_col, vocab_col
from auth import hash_password, verify_password, create_token, get_current_user, require_user
from chains import run_agent_analyze, run_session_analysis, run_tutor_chat, run_poem_generator

app = FastAPI(title="TOEIC AI Agent", version="3.0-mongo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    # allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await create_indexes()


# ── Schemas ───────────────────────────────────────────────────────
class RegisterReq(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=50, description="Mật khẩu từ 6-50 ký tự")
    full_name: Optional[str] = ""

class LoginReq(BaseModel):
    email: EmailStr
    password: str

# class ChatMessage(BaseModel):
#     role: str
#     content: str

class ChatReq(BaseModel):
    question: str
    history: list[dict] = []
    # history: List[ChatMessage] = []

class AnalyzeReq(BaseModel):
    stats: dict

class SessionReq(BaseModel):
    mode: str
    total: int
    correct: int

class PoemReq(BaseModel):
    word: str

class StatsReq(BaseModel):
    mode: str
    done: int
    correct: int


# ── Health ────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    # Ping MongoDB để kiểm tra kết nối
    from database import get_client
    try:
        await get_client().admin.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    return {
        "status": "ok",
        "database": "mongodb_atlas",
        "db_status": db_status,
        "langsmith_project": os.getenv("LANGCHAIN_PROJECT"),
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2"),
    }


# ══ AUTH ══════════════════════════════════════════════════════════

@app.post("/api/auth/register")
async def register(req: RegisterReq):
    if await users_col().find_one({"email": req.email}):
        raise HTTPException(400, "Email đã được sử dụng")

    uid = str(uuid.uuid4())
    await users_col().insert_one({
        "_id":        uid,
        "email":      req.email,
        "password":   hash_password(req.password),
        "full_name":  req.full_name or "",
        "xp":         0,
        "streak":     0,
        "created_at": datetime.now(timezone.utc),
    })
    return {
        "access_token": create_token(uid),
        "token_type": "bearer",
        "user_id": uid,
        "email": req.email,
        "full_name": req.full_name or "",
    }


@app.post("/api/auth/login")
async def login(req: LoginReq):
    user = await users_col().find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(401, "Email hoặc mật khẩu không đúng")

    return {
        "access_token": create_token(user["_id"]),
        "token_type": "bearer",
        "user_id": user["_id"],
        "email": user["email"],
        "full_name": user.get("full_name", ""),
    }


@app.get("/api/me")
async def get_me(user=Depends(require_user)):
    uid = user["_id"]
    stats_docs = await stats_col().find({"user_id": uid}).to_list(None)
    stats = {d["mode"]: {"done": d["done"], "correct": d["correct"]} for d in stats_docs}
    for m in ["reading", "grammar", "vocab", "listening"]:
        stats.setdefault(m, {"done": 0, "correct": 0})
    return {
        "profile": {
            "id":        uid,
            "email":     user["email"],
            "full_name": user.get("full_name", ""),
            "xp":        user.get("xp", 0),
            "streak":    user.get("streak", 0),
        },
        "stats": stats,
    }


# ══ STATS ════════════════════════════════════════════════════════

@app.post("/api/stats/upsert")
async def upsert_stats(req: StatsReq, user=Depends(require_user)):
    uid = user["_id"]
    existing = await stats_col().find_one({"user_id": uid, "mode": req.mode})
    if existing:
        await stats_col().update_one(
            {"user_id": uid, "mode": req.mode},
            {"$inc": {"done": req.done, "correct": req.correct},
             "$set": {"updated_at": datetime.now(timezone.utc)}}
        )
    else:
        await stats_col().insert_one({
            "user_id":    uid,
            "mode":       req.mode,
            "done":       req.done,
            "correct":    req.correct,
            "updated_at": datetime.now(timezone.utc),
        })
    # Cộng XP
    xp_gain = req.correct * 20 + (req.done - req.correct) * 5
    await users_col().update_one({"_id": uid}, {"$inc": {"xp": xp_gain}})
    return {"ok": True}


# ══ AGENT ════════════════════════════════════════════════════════

@app.post("/api/agent/analyze")
async def agent_analyze(req: AnalyzeReq, user=Depends(get_current_user)):
    """Agent phân tích điểm yếu → LangSmith: agent_analyze"""
    stats = req.stats
    labels = {"reading":"Reading","grammar":"Grammar","vocab":"Vocabulary","listening":"Listening"}
    with_data = [m for m in stats if stats[m].get("done", 0) > 0]

    if not with_data:
        return {"message": "Hãy làm bài để Agent theo dõi điểm yếu và đề xuất lộ trình!", "weakest": None}

    summary = ", ".join(
        f"{labels[m]}: {round(stats[m]['correct']/stats[m]['done']*100)}% ({stats[m]['done']} câu)"
        for m in with_data
    )
    weakest = min(with_data, key=lambda m: stats[m]["correct"] / stats[m]["done"])

    try:
        message = await run_agent_analyze(summary)
    except Exception:
        acc = round(stats[weakest]["correct"] / stats[weakest]["done"] * 100)
        message = f"Bạn đang yếu nhất ở {labels[weakest]} ({acc}%). Hãy luyện thêm!"

    return {"message": message, "weakest": weakest}


@app.post("/api/agent/session-result")
async def session_result(req: SessionReq, user=Depends(get_current_user)):
    """Phân tích + lưu session → MongoDB + LangSmith: session_result_analysis"""
    acc = round(req.correct / req.total * 100) if req.total > 0 else 0
    est = 300 + round(acc * 5.95)
    labels = {"reading":"Reading","grammar":"Grammar","vocab":"Vocabulary","listening":"Listening"}

    try:
        message = await run_session_analysis(
            mode=labels.get(req.mode, req.mode),
            total=req.total, correct=req.correct, acc=acc,
        )
    except Exception:
        message = f"Bạn đạt {acc}% – {'Tốt lắm!' if acc >= 70 else 'Cố gắng luyện thêm!'}"

    if user:
        await sessions_col().insert_one({
            "user_id":    user["_id"],
            "mode":       req.mode,
            "total":      req.total,
            "correct":    req.correct,
            "accuracy":   acc,
            "est_score":  est,
            "agent_msg":  message,
            "created_at": datetime.now(timezone.utc),
        })

    return {"message": message, "accuracy": acc, "estimated_score": est}


# ══ CHAT ════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(req: ChatReq, user=Depends(get_current_user)):
    """Chat AI tutor → LangSmith: tutor_chat"""
    try:
        reply = await run_tutor_chat(req.question, req.history)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"text": reply}



# ══ VOCAB ════════════════════════════════════════════════════════

@app.post("/api/vocab/poem")
async def generate_poem(req: PoemReq, user=Depends(get_current_user)):
    """Tạo thơ + lưu MongoDB → LangSmith: vocab_poem_generator"""
    try:
        data = await run_poem_generator(req.word)
    except Exception as e:
        raise HTTPException(500, f"Lỗi tạo thơ: {e}")

    if user:
        await vocab_col().insert_one({
            "user_id":    user["_id"],
            "word":       req.word,
            "meaning":    data.get("meaning"),
            "poem":       data.get("poem"),
            "example":    data.get("example"),
            "created_at": datetime.now(timezone.utc),
        })
    return data


@app.get("/api/vocab/history")
async def vocab_history(user=Depends(require_user)):
    docs = await vocab_col().find(
        {"user_id": user["_id"]}, {"_id": 0}
    ).sort("created_at", -1).limit(20).to_list(None)
    for d in docs:
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
    return {"history": docs}


@app.get("/api/sessions")
async def get_sessions(user=Depends(require_user)):
    docs = await sessions_col().find(
        {"user_id": user["_id"]}, {"_id": 0}
    ).sort("created_at", -1).limit(10).to_list(None)
    for d in docs:
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
    return {"sessions": docs}
