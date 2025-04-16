'use client';

import React, { useEffect } from 'react';
import { FaCheck, FaTimes, FaFire } from 'react-icons/fa';
import styles from './FeedbackCard.module.css'; // Import CSS Module

interface FeedbackCardProps {
  isCorrect: boolean;
  actualLabel: 'phishing' | 'legitimate';
  onNext: () => void;
  autoAdvanceDelay?: number; // in milliseconds
  streakCount?: number; // New prop for tracking correct answer streaks
}

const FeedbackCard: React.FC<FeedbackCardProps> = ({ 
  isCorrect, 
  actualLabel, 
  onNext,
  autoAdvanceDelay = 2000, // 2 seconds default
  streakCount = 0 // Default to 0
}) => {
  // Auto-advance to next email after delay
  useEffect(() => {
    const timer = setTimeout(() => {
      onNext();
    }, autoAdvanceDelay);
    
    return () => clearTimeout(timer);
  }, [onNext, autoAdvanceDelay]);

  // Determine card and icon styles based on correctness
  const cardStyle = isCorrect ? styles.cardCorrect : styles.cardIncorrect;
  const iconBackgroundStyle = isCorrect ? styles.iconBackgroundCorrect : styles.iconBackgroundIncorrect;
  const iconStyle = isCorrect ? styles.iconCorrect : styles.iconIncorrect;
  const headingStyle = isCorrect ? styles.headingCorrect : styles.headingIncorrect;

  return (
    <div className={styles.container}>
      <div className={cardStyle}>
        <div className={styles.iconWrapper}>
          <div className={iconBackgroundStyle}>
            {isCorrect ? <FaCheck className={iconStyle} /> : <FaTimes className={iconStyle} />}
          </div>
        </div>
        
        <h2 className={headingStyle}>
          {isCorrect ? 'Correct!' : 'Oops!'}
        </h2>
        
        <p className={styles.message}>
          That was {actualLabel === 'phishing' ? 'a phishing attempt' : 'a legitimate email'}.
        </p>
        
        {/* Show streak info if correct and has a streak */}
        {isCorrect && streakCount > 1 && (
          <div className={styles.streakMessage}>
            <FaFire className={styles.streakIcon} />
            <span>{streakCount} in a row!</span>
          </div>
        )}
        
        <button
          onClick={onNext}
          className={styles.nextButton}
        >
          Next Email
        </button>
        
        <div className={styles.timerText}>
          Next email in {Math.ceil(autoAdvanceDelay / 1000)}s...
        </div>
      </div>
    </div>
  );
};

export default FeedbackCard; 