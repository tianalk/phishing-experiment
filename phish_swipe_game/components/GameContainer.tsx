'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  Email, 
  UserClassification,
  fetchRandomEmails, 
  recordClassification, 
  generateSessionId,
  isSupabaseConfigured
} from '../lib/supabase';
import EmailCard from './EmailCard';
import FeedbackCard from './FeedbackCard';
import styles from './GameContainer.module.css'; // Import CSS Module

const GameContainer: React.FC = () => {
  // Core game state
  const [emails, setEmails] = useState<Email[]>([]);
  const [currentEmailIndex, setCurrentEmailIndex] = useState<number>(0);
  const [sessionId, setSessionId] = useState<string>('');
  const [score, setScore] = useState<number>(0);
  const [totalClassified, setTotalClassified] = useState<number>(0);
  const [sessionEmailsClamped, setSessionEmailsClamped] = useState<number>(0);

  // UI state
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showFeedback, setShowFeedback] = useState<boolean>(false);
  const [lastClassification, setLastClassification] = useState<'phishing' | 'legitimate' | null>(null);
  const [cardHeight, setCardHeight] = useState<string>('auto');
  const [streak, setStreak] = useState<number>(0); // Track correct answers streak
  const [showTutorial, setShowTutorial] = useState<boolean>(true); // Tutorial state
  
  // Refs
  const cardRef = useRef<HTMLDivElement>(null);
  
  // Config
  const sessionEmailsCount = Number(process.env.NEXT_PUBLIC_SESSION_EMAILS_COUNT || 10);
  
  // Initialize game on component mount
  useEffect(() => {
    const initGame = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Generate a new session ID
        const newSessionId = generateSessionId();
        setSessionId(newSessionId);
        
        // Fetch random emails (uses sample data if not configured)
        const randomEmails = await fetchRandomEmails(sessionEmailsCount);
        if (randomEmails.length === 0) {
          throw new Error('Failed to load emails. Please try again.');
        }
        
        setEmails(randomEmails);
        setSessionEmailsClamped(Math.min(sessionEmailsCount, randomEmails.length));
        setCurrentEmailIndex(0);
        setScore(0);
        setTotalClassified(0);
        setStreak(0);
      } catch (err: any) {
        console.error('Error initializing game:', err);
        setError(err.message || 'Failed to load the game. Please refresh the page.');
      } finally {
        setLoading(false);
      }
    };
    
    initGame();
  }, [sessionEmailsCount]);
  
  // Adjust container height based on card content
  useEffect(() => {
    // Simple function to measure and set height
    function updateHeight() {
      if (cardRef.current) {
        const height = cardRef.current.scrollHeight;
        // Add minimal padding to avoid content being cut off
        setCardHeight(`${height + 30}px`);
      }
    }
    
    // Wait a short time to ensure content is rendered
    const timer = setTimeout(updateHeight, 50);
    
    return () => clearTimeout(timer);
  }, [currentEmailIndex, showFeedback, emails, showTutorial]);
  
  // Handle email classification
  const handleClassify = async (classification: 'phishing' | 'legitimate') => {
    if (currentEmailIndex >= emails.length) return;
    
    // Hide tutorial if still showing
    if (showTutorial) {
      setShowTutorial(false);
    }
    
    const currentEmail = emails[currentEmailIndex];
    const isCorrect = classification === currentEmail.true_label;
    
    // Update score and streak
    if (isCorrect) {
      setScore(prevScore => prevScore + 1);
      setStreak(prevStreak => prevStreak + 1);
    } else {
      // Reset streak on wrong answer
      setStreak(0);
    }
    
    // Prepare classification data for database
    const classificationData: Omit<UserClassification, 'id'> = {
      email_id: currentEmail.id,
      user_classification: classification,
      true_label: currentEmail.true_label,
      source_dataset: currentEmail.source,
      is_correct: isCorrect,
      timestamp: new Date().toISOString(),
      session_id: sessionId
    };
    
    // Store classification ONLY if configured
    if (isSupabaseConfigured) {
      try {
        await recordClassification(classificationData);
      } catch (err) {
        console.error('Error recording classification:', err);
        // Continue game flow even with error
      }
    }
    
    setLastClassification(classification);
    setShowFeedback(true);
    setTotalClassified(totalClassified + 1);
  };
  
  // Handle advancing to next email
  const handleNext = () => {
    setShowFeedback(false);
    setLastClassification(null);
    
    if (currentEmailIndex < emails.length - 1) {
      setCurrentEmailIndex(currentEmailIndex + 1);
    }
  };
  
  // Skip tutorial
  const handleSkipTutorial = () => {
    setShowTutorial(false);
  };
  
  // Game metrics
  const gameProgress = Math.floor((totalClassified / sessionEmailsClamped) * 100);
  const currentEmail = emails[currentEmailIndex];
  const isGameFinished = sessionEmailsClamped > 0 && totalClassified >= sessionEmailsClamped;
  
  // Loading state
  if (loading) {
    return (
      <div className={styles.loadingState}>
        <div className={styles.loadingSpinner}></div>
        <p>Loading emails...</p>
      </div>
    );
  }
  
  // Error state
  if (error) {
    return (
      <div className={styles.errorState}>
        <div className={styles.errorIcon}>!</div>
        <h3>Something went wrong</h3>
        <p>{error}</p>
        <button
          onClick={() => window.location.reload()}
          className={styles.retryButton}
        >
          Try Again
        </button>
      </div>
    );
  }
  
  // Game finished state
  if (isGameFinished) {
    const scorePercentage = Math.round((score / sessionEmailsClamped) * 100);
    
    return (
      <div className={styles.finishedState}>
        <h2>Game Complete!</h2>
        
        <div className={styles.scoreCircle}>
          <div className={styles.scoreValue}>{score}</div>
          <div className={styles.scoreTotal}>/ {sessionEmailsClamped}</div>
        </div>
        
        <div className={styles.scorePercentage}>
          {scorePercentage}% Accuracy
        </div>
        
        <p className={styles.finishedMessage}>
          {scorePercentage === 100 
            ? 'Perfect! You have a keen eye for phishing attempts.' 
            : scorePercentage >= 80 
              ? 'Great job! You\'re well-equipped to spot most phishing attempts.' 
              : scorePercentage >= 60 
                ? 'Good effort! With more practice, you\'ll become even better at identifying phishing.' 
                : 'Keep practicing! Phishing can be tricky to spot.'}
        </p>
        
        {!isSupabaseConfigured && (
           <p className={styles.finishedStaticNote}>
            Note: This game is running in demo mode. Your results were not saved.
           </p>
        )}
        
        <button
          onClick={() => window.location.reload()}
          className={styles.playAgainButton}
        >
          Play Again
        </button>
      </div>
    );
  }
  
  // Active gameplay
  return (
    <div className={styles.container}>
      {/* Static Mode Indicator */}
      {!isSupabaseConfigured && (
        <div className={styles.staticModeIndicator}>
            Demo Mode: Results will not be saved
        </div>
      )}
      
      {/* Game Stats */}
      <div className={styles.statsContainer}>
        <div className={styles.progressContainer}>
          <div className={styles.progressInfo}>
            <span>Email {totalClassified + 1} of {sessionEmailsClamped}</span>
            <span>{score} correct</span>
          </div>
          <div className={styles.progressBarBackground}>
            <div
              className={styles.progressBarFill}
              style={{ width: `${gameProgress}%` }}
            >
              {gameProgress > 10 && (
                <span className={styles.progressBarText}>{gameProgress}%</span>
              )}
            </div>
          </div>
        </div>
        
        {/* Streak indicator - only show if there's a streak */}
        {streak > 1 && (
          <div className={styles.streakIndicator}>
            <span className={styles.streakValue}>{streak}</span> correct in a row!
          </div>
        )}
      </div>
      
      {/* Game Content Area */}
      <div
        className={styles.gameContentArea}
        style={{ height: cardHeight }}
      >
        <div ref={cardRef} className={styles.cardPositioner}>
          {showTutorial ? (
            <div className={styles.tutorial}>
              <h3>How to Play</h3>
              <div className={styles.tutorialPoints}>
                <div className={styles.tutorialPoint}>
                  <div className={styles.tutorialNumber}>1</div>
                  <p>Examine each email carefully for signs of phishing</p>
                </div>
                <div className={styles.tutorialPoint}>
                  <div className={styles.tutorialNumber}>2</div>
                  <p>Swipe right if you think it's a phishing attempt</p>
                </div>
                <div className={styles.tutorialPoint}>
                  <div className={styles.tutorialNumber}>3</div>
                  <p>Swipe left if you think it's legitimate</p>
                </div>
              </div>
              <button
                onClick={handleSkipTutorial}
                className={styles.tutorialButton}
              >
                Start Game
              </button>
            </div>
          ) : showFeedback ? (
            <FeedbackCard 
              key={`feedback-${currentEmailIndex}`}
              isCorrect={lastClassification === currentEmail.true_label}
              actualLabel={currentEmail.true_label}
              onNext={handleNext}
              streakCount={streak}
            />
          ) : (
            currentEmail && (
              <EmailCard 
                key={currentEmail.id}
                email={currentEmail} 
                onClassify={handleClassify} 
                isLastCard={currentEmailIndex === emails.length - 1}
              />
            )
          )}
        </div>
      </div>
      
      {/* Hint text - shows during active gameplay */}
      {!showTutorial && !showFeedback && (
        <div className={styles.hint}>
          Look for suspicious links, urgent language, spelling errors, and requests for personal information
        </div>
      )}
    </div>
  );
};

export default GameContainer; 