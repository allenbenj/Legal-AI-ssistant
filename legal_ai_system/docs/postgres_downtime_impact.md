# PostgreSQL Downtime Impact

The system is designed to gracefully degrade when the PostgreSQL service is unavailable. This document outlines what functionality is affected and what remains operational.

## Completely Unavailable

- **User Authentication & Security**
  - Login/logout
  - Session persistence
  - Access control
  - Audit logging
- **Knowledge Management**
  - Entity extraction storage
  - Relationship mapping
  - Knowledge graph
  - Cross-document linking
- **Advanced Search & AI**
  - Vector metadata
  - Content deduplication
  - Search optimization
  - Enhanced search results
- **Workflow Management**
  - Workflow persistence
  - Progress tracking
  - Batch processing

## Degraded Features

- **Document Processing**
  - Text extraction works
  - AI analysis works
  - Cannot save results persistently
  - No processing status tracking
- **Search**
  - Basic FAISS vector search works
  - No metadata enhancement
  - No search history

## Fully Functional (SQLite-based)

- Core document processing
- Local memory storage
- Basic GUI
- FAISS vector operations
- Local database for violation tracking

## User Experience Impact

With PostgreSQL offline the application behaves like a single-user session-based tool. Documents can still be processed and analysed, but persistent knowledge management and multi-user collaboration are not available.

