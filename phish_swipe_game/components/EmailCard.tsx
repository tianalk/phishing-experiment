'use client';

import React, { useState, useEffect } from 'react';
import { useSwipeable, SwipeEventData } from 'react-swipeable';
import { FaCheck, FaTimes, FaArrowLeft, FaArrowRight, FaEnvelope, FaExclamationTriangle, FaShieldAlt } from 'react-icons/fa';
import { Email } from '../lib/supabase';
import styles from './EmailCard.module.css';

interface EmailCardProps {
  email: Email;
  onClassify: (classification: 'phishing' | 'legitimate') => void;
  isLastCard?: boolean;
}

const EmailCard: React.FC<EmailCardProps> = ({ email, onClassify, isLastCard = false }) => {
  const [swipeDirection, setSwipeDirection] = useState<string | null>(null);
  const [isSwiping, setIsSwiping] = useState<boolean>(false);
  const [swipeProgress, setSwipeProgress] = useState<number>(0);
  const [currentDeltaX, setCurrentDeltaX] = useState<number>(0);
  const [currentDir, setCurrentDir] = useState<'Left' | 'Right' | 'Up' | 'Down' | null>(null);
  const [showHints, setShowHints] = useState<boolean>(false);

  // Extract sender and subject from email body if available
  const emailLines = email.email_body.split('\n');
  const sender = emailLines.find(line => line.toLowerCase().includes('from:'))?.replace('From:', '').trim() || 'Unknown Sender';
  const subject = emailLines.find(line => line.toLowerCase().includes('subject:'))?.replace('Subject:', '').trim() || 'No Subject';

  // Auto-detect potential phishing indicators
  const hasUrgentLanguage = email.email_body.toLowerCase().match(/urgent|immediate|now|alert|verify|suspend|disabled|limited time/g);
  const hasUnusualLinks = email.email_body.toLowerCase().includes('click here') || 
                          email.email_body.includes('http') ||
                          email.email_body.match(/\[.*?\]\(.*?\)/g); // Markdown links
  const requestsPersonalInfo = email.email_body.toLowerCase().match(/password|credit card|ssn|social security|bank account|verify.*account|confirm.*information/g);
  
  // Show auto-detected hints after a short delay if this isn't the last card
  useEffect(() => {
    if (!isLastCard) {
      const timer = setTimeout(() => {
        setShowHints(true);
      }, 5000); // Show hints after 5 seconds
      
      return () => clearTimeout(timer);
    }
  }, [isLastCard]);

  // Configure swipe handlers
  const handlers = useSwipeable({
    onSwipedLeft: () => handleSwipe('legitimate'),
    onSwipedRight: () => handleSwipe('phishing'),
    onSwiping: (eventData: SwipeEventData) => {
      setIsSwiping(true);
      setCurrentDeltaX(eventData.deltaX);
      setCurrentDir(eventData.dir);
      const progress = Math.abs(eventData.deltaX) / (window.innerWidth * 0.3);
      setSwipeProgress(Math.min(progress, 1));
    },
    onSwiped: () => {
      // If swipe wasn't strong enough, return to center
      if (swipeProgress < 0.5) {
        setIsSwiping(false);
        setSwipeProgress(0);
        setCurrentDeltaX(0);
        setCurrentDir(null);
      }
    },
    trackMouse: true,
    preventScrollOnSwipe: true,
    delta: 10, // More sensitive swipe detection
    swipeDuration: 500, // Slightly longer swipe duration for better detection
  });

  const handleSwipe = (classification: 'phishing' | 'legitimate') => {
    setSwipeDirection(classification);
    setIsSwiping(false);
    
    // Animate the card off-screen
    setTimeout(() => {
      onClassify(classification);
      setSwipeDirection(null);
      setSwipeProgress(0);
      setCurrentDeltaX(0);
      setCurrentDir(null);
    }, 300);
  };

  const handleButtonClick = (classification: 'phishing' | 'legitimate') => {
    handleSwipe(classification);
  };

  // Calculate dynamic styles based on swipe state
  const swipingClass = isSwiping && swipeProgress > 0.1
    ? currentDir === 'Left' 
      ? styles.swipingLeft 
      : currentDir === 'Right' 
        ? styles.swipingRight 
        : ''
    : '';

  // Determine transform and color changes based on swipe direction and progress
  const cardTransformStyle = {
    transform: swipeDirection === 'phishing' 
      ? 'translateX(150px) rotate(20deg)'
      : swipeDirection === 'legitimate' 
      ? 'translateX(-150px) rotate(-20deg)'
      : isSwiping 
      ? `translateX(${currentDeltaX}px) rotate(${currentDeltaX / 20}deg)`
      : 'translateX(0) rotate(0)',
    opacity: swipeDirection ? 0 : 1,
    transition: isSwiping ? 'none' : 'transform 0.3s ease-out, opacity 0.3s ease-out',
  };

  // Overlay opacity based on swipe progress
  const leftOverlayOpacity = currentDir === 'Left' ? Math.min(swipeProgress * 1.5, 1) : 0;
  const rightOverlayOpacity = currentDir === 'Right' ? Math.min(swipeProgress * 1.5, 1) : 0;

  return (
    <div
      {...handlers}
      className={`${styles.swipeContainer} ${isSwiping ? styles.swiping : ''} ${swipingClass}`}
      style={cardTransformStyle}
    >
      <div className={styles.card}>
        {/* Swipe decision overlays */}
        <div 
          className={styles.swipeOverlayLeft}
          style={{ opacity: leftOverlayOpacity }}
        >
          <FaCheck className={styles.overlayIcon} /> Real
        </div>
        <div 
          className={styles.swipeOverlayRight}
          style={{ opacity: rightOverlayOpacity }}
        >
          Phish <FaExclamationTriangle className={styles.overlayIcon} />
        </div>

        {/* Email header */}
        <div className={styles.emailHeader}>
          <div className={styles.emailIcon}>
            <FaEnvelope />
          </div>
          <div className={styles.emailDetails}>
            <div className={styles.emailSubject}>{subject}</div>
            <div className={styles.emailSender}>{sender}</div>
          </div>
        </div>

        {/* Email content */}
        <div className={styles.emailSection}>
          <div className={styles.emailContent}>
            {email.email_body.split('\n').map((line, i) => (
              <React.Fragment key={i}>
                {line}
                {i < email.email_body.split('\n').length - 1 && <br />}
              </React.Fragment>
            ))}
          </div>

          {/* Phishing indicators */}
          {showHints && (hasUrgentLanguage || hasUnusualLinks || requestsPersonalInfo) && (
            <div className={styles.phishingIndicators}>
              <div className={styles.indicatorHeader}>
                <FaShieldAlt /> Potential indicators detected:
              </div>
              <ul className={styles.indicatorList}>
                {hasUrgentLanguage && (
                  <li>Contains urgent language or pressure tactics</li>
                )}
                {hasUnusualLinks && (
                  <li>Contains links or "click here" instructions</li>
                )}
                {requestsPersonalInfo && (
                  <li>Requests sensitive personal information</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* Control buttons */}
        <div className={styles.controlsSection}>
          <div className={styles.buttonContainer}>
            <button
              onClick={() => handleButtonClick('legitimate')}
              className={styles.realButton}
              aria-label="Classify as Real (Legitimate)"
            >
              <FaCheck /> Real
            </button>
            <button
              onClick={() => handleButtonClick('phishing')}
              className={styles.phishButton}
              aria-label="Classify as Phishing"
            >
              <FaTimes /> Phish
            </button>
          </div>

          <div className={styles.instructions}>
            Swipe left = Real / Swipe right = Phish
          </div>
        </div>

      </div>
    </div>
  );
};

export default EmailCard; 