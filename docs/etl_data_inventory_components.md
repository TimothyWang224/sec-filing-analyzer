# ETL Data Inventory Components Classification

This document classifies the components of the ETL Data Inventory page as either using real data or simulated data.

## Data Classification Table

| Feature/Component | Data Type | Classification | Notes |
|-------------------|-----------|----------------|-------|
| **Preview Selection** |
| Selected Companies | User Selection | **Real** | Shows the actual companies selected by the user |
| Selected Filing Types | User Selection | **Real** | Shows the actual filing types selected by the user |
| Date Range | User Selection | **Real** | Shows the actual date range selected by the user |
| Estimated Filings | Calculated | **Simulated** | Based on a formula using companies, filing types, and date range |
| Estimated Storage | Calculated | **Simulated** | Based on estimated filings × 2MB per filing |
| Estimated Time | Calculated | **Simulated** | Based on estimated filings ÷ 10 filings per minute |
| **ETL Terminal Output** |
| Companies Processed | Sample Data | **Simulated** | Uses predefined sample companies, not user selection |
| Filing Types Processed | Sample Data | **Simulated** | Uses predefined sample filing types |
| Filing Dates | Sample Data | **Simulated** | Uses predefined sample dates |
| File Sizes | Random Generation | **Simulated** | Random numbers between 500-5000KB |
| Data Points Extracted | Random Generation | **Simulated** | Random numbers between 50-500 |
| XBRL Tags Processed | Random Generation | **Simulated** | Random numbers between 20-200 |
| Records Stored | Random Generation | **Simulated** | Random numbers between 10-100 |
| Embeddings Generated | Random Generation | **Simulated** | Random numbers between 5-50 |
| Progress Stages | Predefined | **Simulated** | Uses 10 predefined stages regardless of actual work |
| Progress Percentage | Predefined | **Simulated** | Increments at predetermined rates |
| **Job Tracking** |
| Job ID | Generated | **Simulated** | Sequential ID based on existing sample jobs |
| Job Status | Static | **Simulated** | Always shows "Completed" |
| Companies in Job | User Selection | **Real** | Uses the actual companies selected by user |
| Filing Types in Job | User Selection | **Real** | Uses the actual filing types selected by user |
| Date Range in Job | User Selection | **Real** | Uses the actual date range selected by user |
| Filings Count in Job | Calculated | **Simulated** | Uses the same estimate as Preview Selection |
| Submission Time | System Time | **Real** | Uses the actual current time |
| **Recent ETL Jobs** |
| All Job Entries | Sample Data | **Simulated** | Predefined sample jobs with static data |

## Implementation Notes

In a production environment, all simulated components would be replaced with real data:

1. **Estimates**: Would be based on actual database queries and historical ETL performance
2. **Terminal Output**: Would stream real-time logs from the actual ETL process
3. **Job Tracking**: Would use a real job queue and database for persistence
4. **Recent Jobs**: Would display actual historical jobs from the database

The current implementation provides a realistic simulation of what the real functionality would look like, but it's using sample data and formulas rather than connecting to the actual ETL pipeline and database.

## Future Enhancements

Potential enhancements to make the simulation more realistic:

1. Add a note indicating which components are simulations
2. Use the actual selected companies in the terminal output simulation
3. Add more realistic error handling and recovery scenarios
4. Implement actual database queries to show what's already in the system
5. Add a "dry run" mode that shows what would be retrieved without actually running the ETL
