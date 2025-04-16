-- This SQL file defines the database schema for the Phish Swipe game
-- You can run this in the Supabase SQL Editor to set up your tables

-- Email dataset table
CREATE TABLE emails (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email_body TEXT NOT NULL,
  true_label TEXT NOT NULL CHECK (true_label IN ('phishing', 'legitimate')),
  source TEXT NOT NULL CHECK (source IN ('human', 'ai')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster random selection
CREATE INDEX emails_source_idx ON emails(source);

-- User classification results table
CREATE TABLE user_classifications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email_id UUID REFERENCES emails(id),
  session_id TEXT NOT NULL,
  user_classification TEXT NOT NULL CHECK (user_classification IN ('phishing', 'legitimate')),
  true_label TEXT NOT NULL CHECK (true_label IN ('phishing', 'legitimate')),
  source_dataset TEXT NOT NULL CHECK (source_dataset IN ('human', 'ai')),
  is_correct BOOLEAN NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for analysis queries
CREATE INDEX user_classifications_session_idx ON user_classifications(session_id);
CREATE INDEX user_classifications_source_idx ON user_classifications(source_dataset);
CREATE INDEX user_classifications_correct_idx ON user_classifications(is_correct);

-- Sample query to get email classification statistics by source
-- SELECT 
--   source_dataset,
--   COUNT(*) as total_classifications,
--   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_classifications,
--   ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
-- FROM 
--   user_classifications
-- GROUP BY 
--   source_dataset;

-- Sample query to find sessions with high classification accuracy
-- SELECT 
--   session_id,
--   COUNT(*) as total_classifications,
--   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_classifications,
--   ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
-- FROM 
--   user_classifications
-- GROUP BY 
--   session_id
-- HAVING 
--   COUNT(*) >= 5
-- ORDER BY 
--   accuracy_percentage DESC;

-- Sample import statement - you would replace this with actual data import
-- INSERT INTO emails (email_body, true_label, source)
-- VALUES 
--  ('Dear user, your account has been suspended. Click here to reactivate.', 'phishing', 'human'),
--  ('Your order #12345 has been shipped and will arrive on Friday, July 7th.', 'legitimate', 'human'),
--  ('Urgent: Your bank account has been compromised. Please verify your identity by clicking on this link.', 'phishing', 'ai'),
--  ('Thank you for your recent purchase. Your receipt is attached.', 'legitimate', 'ai'); 