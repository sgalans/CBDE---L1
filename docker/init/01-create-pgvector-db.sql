-- Runs only when the data volume is empty (first `docker compose up`).
-- Creates the separate database used by the optional pgvector part (G0-G2).
-- The extension itself is enabled by G0 inside this database, never in `cbde`.
CREATE DATABASE cbde_pgvector OWNER cbde;
