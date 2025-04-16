import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Define the type for email data
export type Email = {
  id: string;
  email_body: string;
  true_label: 'phishing' | 'legitimate';
  source: 'human' | 'ai';
};

// Define the type for user classification data
export type UserClassification = {
  id?: string;
  email_id: string;
  user_classification: 'phishing' | 'legitimate';
  true_label: 'phishing' | 'legitimate';
  source_dataset: 'human' | 'ai';
  is_correct: boolean;
  timestamp: string;
  session_id: string;
};

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

// Flag to check if Supabase is configured
export const isSupabaseConfigured = !!(supabaseUrl && supabaseAnonKey);

let supabase: SupabaseClient;

if (isSupabaseConfigured) {
  supabase = createClient(supabaseUrl, supabaseAnonKey);
  console.log('Supabase client initialized.');
} else {
  console.warn('Supabase environment variables not set. Running in static mode.');
  // Create a placeholder object if needed, or handle calls conditionally
  supabase = {} as SupabaseClient; // Assign a dummy object to satisfy type checks
}

// Sample data for static mode
const sampleEmails: Email[] = [
  {
    id: 'sample-1',
    email_body: 'Dear Valued Customer,\n\nYour account requires immediate attention due to suspicious activity. Please click the link below to verify your identity and secure your account.\n\n[Suspicious Link Removed]\n\nFailure to comply within 24 hours will result in account suspension.\n\nSincerely,\nYour Bank Security Team',
    true_label: 'phishing',
    source: 'human'
  },
  {
    id: 'sample-2',
    email_body: 'Hi Team,\n\nJust a reminder about our weekly sync meeting tomorrow at 10 AM PST. The agenda includes a review of Q2 goals and brainstorming for the upcoming project.\n\nPlease find the meeting link below:\n[Meeting Link Removed]\n\nSee you there!\n\nBest,\nProject Manager',
    true_label: 'legitimate',
    source: 'human'
  },
  {
    id: 'sample-3',
    email_body: 'CONGRATULATIONS! You have been selected to receive a $1000 gift card. Click here now to claim your prize! Limited time offer!\n\n[Malicious Link Removed]\n\nAct fast before it expires!',
    true_label: 'phishing',
    source: 'ai'
  },
  {
    id: 'sample-4',
    email_body: 'Your recent order #XYZ789 has shipped!\n\nEstimated delivery date: 3-5 business days.\nYou can track your package here:\n[Tracking Link Removed]\n\nThank you for shopping with us!\n\nCustomer Support',
    true_label: 'legitimate',
    source: 'ai'
  },
    // Add more sample emails as desired
];

// Shuffle array helper
function shuffleArray<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// Helper function to fetch a batch of random emails from both datasets
export async function fetchRandomEmails(count: number = 10): Promise<Email[]> {
  if (!isSupabaseConfigured) {
    console.log('Using sample email data.');
    return shuffleArray([...sampleEmails]).slice(0, count);
  }
  
  try {
    console.log('Fetching emails from Supabase...');
    // Fetch random emails from both human and AI sources
    // Using a View or RPC might be more efficient for random selection at scale
    const { data, error } = await supabase
      .from('emails') // Assuming 'emails' is your table name
      .select('id, email_body, true_label, source')
      .limit(count * 5); // Fetch more than needed to simulate randomness
    
    if (error) {
      console.error('Error fetching emails:', error);
      return [];
    }
    
    // Shuffle and take the required count
    return shuffleArray(data as Email[]).slice(0, count);
  } catch (err) {
    console.error('Error in fetchRandomEmails:', err);
    return [];
  }
}

// Helper function to record a user's classification
export async function recordClassification(classification: Omit<UserClassification, 'id'>): Promise<boolean> {
  if (!isSupabaseConfigured) {
    console.log('Static mode: Skipping classification recording.');
    return true; // Simulate success
  }
  
  try {
    console.log('Recording classification to Supabase...');
    const { error } = await supabase
      .from('user_classifications') // Assuming this is your results table
      .insert([classification]);
    
    if (error) {
      console.error('Error recording classification:', error);
      return false;
    }
    
    console.log('Classification recorded successfully.');
    return true;
  } catch (err) {
    console.error('Error in recordClassification:', err);
    return false;
  }
}

// Helper function to generate a session ID
export function generateSessionId(): string {
  return Math.random().toString(36).substring(2, 15) + 
         Math.random().toString(36).substring(2, 15);
} 