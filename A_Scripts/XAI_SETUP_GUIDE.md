# XAI (Grok) Integration Setup Guide

This guide explains how to set up and use the XAI (xAI/Grok) integration in the Legal AI System GUI.

## 🤖 What is XAI Integration?

The GUI now includes direct integration with xAI's Grok models, allowing you to:
- Process legal documents using Grok-3-Mini and Grok-3-Reasoning
- Perform legal analysis, violation detection, and entity extraction
- Use specialized legal prompts optimized for Grok models
- Compare results between different Grok model variants

## 🚀 Quick Setup

### 1. Get XAI API Key

1. Sign up at [x.ai](https://x.ai) or [console.x.ai](https://console.x.ai)
2. Create an API key in your XAI dashboard
3. Copy your API key (starts with `xai-`)

### 2. Configure in GUI

1. Start the Legal AI System GUI:
   ```bash
   streamlit run legal_ai_system/gui/main_gui.py
   ```

2. Navigate to the **🤖 XAI Document Processor** tab

3. Enter your XAI API key in the configuration section

4. Click **🔍 Test XAI Connection** to verify setup

5. Select your preferred Grok model

## 📋 Available Grok Models

### Grok-3-Mini
- **Use Case**: Fast, efficient legal analysis
- **Context**: 8,192 tokens
- **Best For**: Quick document reviews, basic violation detection
- **Speed**: Fastest response times
- **Cost**: Most economical

### Grok-3-Reasoning  
- **Use Case**: Complex legal reasoning and analysis
- **Context**: 8,192 tokens  
- **Best For**: In-depth legal analysis, complex constitutional issues
- **Speed**: Slower but more thorough
- **Cost**: Higher due to reasoning capabilities

### Grok-2-1212
- **Use Case**: Balanced performance and reasoning
- **Context**: 8,192 tokens
- **Best For**: General legal work, balanced speed/quality
- **Speed**: Moderate
- **Cost**: Moderate

## 🎯 Analysis Types Available

### 1. Legal Analysis
- Comprehensive document review
- Constitutional concerns identification
- Procedural issue detection
- Factual inconsistency analysis

### 2. Violation Detection
- Constitutional violations
- Procedural violations
- Ethical violations
- Evidence handling issues

### 3. Entity Extraction
- People (names, roles, titles)
- Organizations (companies, agencies, courts)
- Locations (addresses, jurisdictions)
- Dates (important dates, deadlines)
- Legal concepts (laws, regulations, cases)
- Financial information (amounts, damages)

### 4. Document Summary
- Case/document overview
- Key legal issues
- Main arguments
- Conclusions and holdings
- Legal implications

## ⚙️ Configuration Options

### Model Parameters
- **Temperature**: Controls creativity vs. focus (0.0-1.0)
  - Legal Analysis: 0.3 (focused)
  - Citation Formatting: 0.1 (very precise)
  - Summaries: 0.5 (balanced)
  - Reasoning: 0.2 (logical)

### Rate Limits
- **Requests per minute**: 60 (adjust based on your plan)
- **Tokens per minute**: 100,000
- **Retry attempts**: 3
- **Timeout**: 60 seconds

## 💡 Usage Tips

### Document Processing
1. **File Types**: Supports PDF, DOCX, TXT, MD files
2. **Size Limits**: Optimized for documents under 10MB
3. **Context Management**: Automatically handles large documents

### Model Selection
- Use **Grok-3-Mini** for quick analysis and batch processing
- Use **Grok-3-Reasoning** for complex legal issues requiring deep analysis
- Use **Grok-2-1212** for balanced performance

### Analysis Optimization
- Select multiple analysis types for comprehensive review
- Adjust temperature based on task:
  - Lower (0.1-0.3) for factual analysis
  - Higher (0.5-0.7) for creative legal arguments
- Use reasoning models for constitutional issues

## 🔒 Security & Privacy

### API Key Security
- Keys are stored in session state (not persisted)
- Recommend using environment variables for production
- Never share or commit API keys to version control

### Data Privacy
- Documents sent to XAI servers for processing
- Review XAI's privacy policy for data handling
- Consider using local models for sensitive documents

### Environment Variables
```bash
# Set XAI API key as environment variable
export XAI_API_KEY="your_xai_api_key_here"
export XAI_MODEL="grok-3-mini"  # Default model
export XAI_BASE_URL="https://api.x.ai/v1"
```

## 🛠️ Troubleshooting

### Common Issues

#### "No API key configured"
- **Solution**: Enter your XAI API key in the configuration section
- **Check**: Ensure the key starts with `xai-`

#### "API connection failed"
- **Check**: Internet connectivity
- **Verify**: API key is valid and active
- **Confirm**: XAI service status

#### "Request rate limit exceeded"
- **Solution**: Wait before making more requests
- **Adjust**: Lower concurrent requests in settings
- **Upgrade**: Consider higher-tier XAI plan

#### "Context length exceeded"
- **Solution**: Use document summarization first
- **Split**: Break large documents into sections
- **Optimize**: Use more focused analysis types

### Error Handling
- All API errors are caught and displayed clearly
- Failed requests automatically retry (up to 3 times)
- Fallback options available for connection issues

## 📊 Monitoring & Analytics

### Token Usage Tracking
- Input tokens used
- Output tokens generated
- Total tokens per request
- Cost estimation (if available)

### Performance Metrics
- Response times by model
- Success/failure rates
- Analysis completion times
- Model comparison data

## 🔄 Integration with Existing Features

### Database Storage
- XAI analysis results stored in local database
- Violation records with XAI confidence scores
- Memory entries from XAI entity extraction
- Full analysis history and audit trail

### Export Options
- JSON export of complete analysis results
- Violation reports with XAI findings
- Entity extraction data for knowledge graph
- Analysis comparison across models

## 🎓 Best Practices

### Model Selection Strategy
1. **Quick Review**: Grok-3-Mini for initial document triage
2. **Deep Analysis**: Grok-3-Reasoning for complex legal issues
3. **Balanced Review**: Grok-2-1212 for general legal work
4. **Comparison**: Use multiple models for critical documents

### Prompt Optimization
- Legal prompts are pre-optimized for each model
- Reasoning models get enhanced step-by-step prompts
- Context automatically adjusted for document size
- Temperature optimized for analysis type

### Cost Management
- Monitor token usage in analysis results
- Use Mini model for bulk processing
- Reserve Reasoning model for complex cases
- Batch similar documents for efficiency

## 📞 Support

### Documentation
- XAI API Documentation: [docs.x.ai](https://docs.x.ai)
- Grok Model Documentation: Available in XAI console
- Legal AI System Docs: See main project documentation

### Troubleshooting Steps
1. Check XAI service status
2. Verify API key configuration
3. Test with simple document first
4. Review error messages in GUI
5. Check network connectivity

### Community
- XAI Community Forums
- Legal AI System GitHub Issues
- Grok Model Updates and Announcements

---

## 🚀 Getting Started Checklist

- [ ] Obtain XAI API key from x.ai
- [ ] Install GUI dependencies: `pip install streamlit pandas plotly`
- [ ] Start GUI: `streamlit run legal_ai_system/gui/main_gui.py`
- [ ] Navigate to 🤖 XAI Document Processor tab
- [ ] Enter API key and test connection
- [ ] Select Grok model (start with Grok-3-Mini)
- [ ] Upload test document
- [ ] Select analysis types
- [ ] Run analysis and review results
- [ ] Export results if needed

**You're now ready to use XAI/Grok models for legal document analysis! 🎉**