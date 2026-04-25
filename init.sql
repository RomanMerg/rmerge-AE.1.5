-- Interview Practice App — Database Schema
-- Run against the 'interview_app' database

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    raw_text TEXT NOT NULL,
    parsed_data JSONB,
    embedding vector(256),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS job_descriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    raw_text TEXT NOT NULL,
    parsed_data JSONB,
    embedding vector(256),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gap_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    cv_id UUID REFERENCES cv_profiles(id),
    jd_id UUID REFERENCES job_descriptions(id),
    analysis JSONB,
    readiness_score INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    model_used VARCHAR(100),
    cost_usd DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    evaluation JSONB,
    overall_score INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity indexes (for dedup in Session 4)
-- Using HNSW instead of IVFFlat — no training needed, works with small datasets
CREATE INDEX IF NOT EXISTS idx_cv_embedding ON cv_profiles USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_jd_embedding ON job_descriptions USING hnsw (embedding vector_cosine_ops);
