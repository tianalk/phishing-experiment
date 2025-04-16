#!/usr/bin/env node

/**
 * CSV to Supabase Importer Script
 * 
 * This script reads CSV files of human-generated and AI-generated emails
 * and imports them into the Supabase 'emails' table for the Phish Swipe game.
 * 
 * Usage:
 * node import_data.js --human=path/to/human_emails.csv --ai=path/to/ai_emails.csv
 * 
 * CSV format should have the columns:
 * - Email_ID
 * - Email_Body
 * - True_Label
 * 
 * Environment Variables:
 * - SUPABASE_URL: Your Supabase project URL
 * - SUPABASE_SERVICE_KEY: Your Supabase service role key (not the anon key)
 */

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { createClient } = require('@supabase/supabase-js');
const dotenv = require('dotenv');
const minimist = require('minimist');

// Load environment variables from .env.local
dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });

// Parse command line arguments
const args = minimist(process.argv.slice(2));

// Check for required files
if (!args.human || !args.ai) {
  console.error('Please provide paths to both human and AI email CSV files.');
  console.error('Usage: node import_data.js --human=path/to/human_emails.csv --ai=path/to/ai_emails.csv');
  process.exit(1);
}

// Initialize Supabase client with service role key for admin operations
const supabaseUrl = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing Supabase environment variables. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Helper function to read CSV file and parse data
function readCsvFile(filePath, source) {
  return new Promise((resolve, reject) => {
    const results = [];
    
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => {
        // Transform CSV row to match database schema
        const transformed = {
          // We don't use the CSV Email_ID as we'll generate UUIDs in the database
          email_body: data.Email_Body,
          true_label: data.True_Label.toLowerCase().trim(),
          source: source // 'human' or 'ai'
        };
        
        // Validate data
        if (!transformed.email_body || !transformed.true_label) {
          console.warn('Skipping row with missing data:', data);
          return;
        }
        
        // Ensure the label is valid
        if (!['phishing', 'legitimate'].includes(transformed.true_label)) {
          console.warn(`Skipping row with invalid label "${transformed.true_label}"`, data);
          return;
        }
        
        results.push(transformed);
      })
      .on('end', () => {
        resolve(results);
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

// Import data function
async function importData() {
  try {
    console.log('Reading human-generated emails CSV...');
    const humanEmails = await readCsvFile(args.human, 'human');
    console.log(`Read ${humanEmails.length} human-generated emails.`);
    
    console.log('Reading AI-generated emails CSV...');
    const aiEmails = await readCsvFile(args.ai, 'ai');
    console.log(`Read ${aiEmails.length} AI-generated emails.`);
    
    // Combine datasets
    const allEmails = [...humanEmails, ...aiEmails];
    
    // Optional: First clear the existing data if needed
    if (args.clear) {
      console.log('Clearing existing email data...');
      const { error: clearError } = await supabase.from('emails').delete().neq('id', '00000000-0000-0000-0000-000000000000');
      if (clearError) {
        console.error('Error clearing data:', clearError);
        process.exit(1);
      }
      console.log('Existing data cleared.');
    }
    
    // Import the data
    console.log('Importing emails to Supabase...');
    const { data, error } = await supabase.from('emails').insert(allEmails);
    
    if (error) {
      console.error('Error importing data:', error);
      process.exit(1);
    }
    
    console.log(`Successfully imported ${allEmails.length} emails.`);
    console.log('Import complete!');
    
  } catch (error) {
    console.error('Error during import process:', error);
    process.exit(1);
  }
}

// Run the import
importData(); 