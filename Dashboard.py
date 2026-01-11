import streamlit as st
import streamlit.components.v1 as components
import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient
import google.generativeai as genai
from openai import OpenAI
from utils import save_search

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="CounterPoint",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for persistence
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'x_intel_result' not in st.session_state:
    st.session_state.x_intel_result = None

def get_xai_client():
    """Create xAI (Grok) client with custom base URL."""
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        return None
    return OpenAI(
        api_key=xai_key,
        base_url="https://api.x.ai/v1"
    )

def get_x_intel(topic):
    """Fetch X.com intelligence using Grok API."""
    client = get_xai_client()
    if not client:
        return {"error": "XAI_API_KEY not configured"}
    
    try:
        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {
                    "role": "system",
                    "content": "You are an investigative journalist tool with access to real-time X (Twitter) data. Analyze the latest posts about the user's topic."
                },
                {
                    "role": "user",
                    "content": f"""Find the latest viral posts and sentiment regarding '{topic}' on X. Return a JSON object with this structure:
{{
   "x_summary": "2 sentence summary of what people are saying on X",
   "viral_rumors": ["Rumor 1", "Rumor 2"],
   "sources": [
      {{"handle": "@username", "link": "https://x.com/user/status/xxx", "text": "snippet of tweet"}}
   ]
}}

IMPORTANT: Return ONLY the JSON object, no other text."""
                }
            ]
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Grok response: {e}", "raw": content}
    except Exception as e:
        return {"error": f"Error calling Grok API: {e}"}

def get_live_news(topic):
    """Fetches live news using Tavily API. Prioritizes recent sources (< 1 year)."""
    tavily_key = os.getenv("Tavily API Key") or os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return None
    
    client = TavilyClient(api_key=tavily_key)
    
    try:
        # First, try to get recent results (within 1 year)
        response = client.search(
            query=topic,
            search_depth="advanced",
            include_domains=[],
            max_results=10,
            time_range="year"  # Only results from past year
        )
        news_data = []
        for result in response.get("results", []):
            news_data.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "title": result.get("title")
            })
        
        # If not enough recent results, search without time filter
        if len(news_data) < 3:
            response = client.search(
                query=topic,
                search_depth="advanced",
                include_domains=[],
                max_results=10
            )
            news_data = []
            for result in response.get("results", []):
                news_data.append({
                    "url": result.get("url"),
                    "content": result.get("content"),
                    "title": result.get("title")
                })
        
        return news_data
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def verify_news(topic, news_data):
    """Verifies news using Gemini API."""
    gemini_key = os.getenv("Gemini API Key") or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return "Error: Gemini API Key missing."

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""
    You are a professional news verification analyst. Analyze these search results about '{topic}'.
    
    Provide a structured report with:
    1. KEY FINDINGS: 3-5 bullet points of verified facts
    2. UNVERIFIED CLAIMS: Any rumors or unverified information found
    3. CONFIDENCE SCORE: A number from 0-100 based on source reliability
    
    Format your response EXACTLY as:
    CONFIDENCE: [number]
    
    KEY FINDINGS:
    • [finding 1]
    • [finding 2]
    • [finding 3]
    
    UNVERIFIED:
    • [claim 1] - [reason why unverified]
    
    SUMMARY:
    [2-3 sentence conclusion]

    Context:
    {news_data}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {e}"

def parse_gemini_to_html(gemini_response, topic):
    """Parse Gemini response into styled HTML sections."""
    lines = gemini_response.split('\n')
    
    # Initialize sections
    confidence = "N/A"
    key_findings = []
    unverified = []
    summary = ""
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if line.startswith('CONFIDENCE:'):
            confidence = line.replace('CONFIDENCE:', '').strip().replace('%', '')
        elif 'KEY FINDINGS' in line.upper():
            current_section = 'findings'
        elif 'UNVERIFIED' in line.upper():
            current_section = 'unverified'
        elif 'SUMMARY' in line.upper():
            current_section = 'summary'
        elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
            bullet_text = line.lstrip('•-* ').strip()
            if current_section == 'findings':
                key_findings.append(bullet_text)
            elif current_section == 'unverified':
                unverified.append(bullet_text)
        elif current_section == 'summary':
            summary += line + " "
    
    # Build styled HTML with embedded font for iframe
    html = f'''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * {{ font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: transparent; }}
</style>
<div style="background: #FFFFFF; padding: 30px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); font-family: 'Inter', sans-serif;">
    
    <div style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 700; font-size: 14px; background: #F0FDF4; color: #166534; margin-bottom: 15px;">
        ✓ CONFIRMED FACTS
    </div>
    <div style="background: #F0FDF4; border-left: 3px solid #3D8B52; padding: 15px 20px; margin: 10px 0 25px 0; border-radius: 6px;">
        <ul style="margin: 0; padding-left: 20px; color: #166534; list-style-type: disc;">
'''
    for finding in key_findings[:5]:
        html += f'            <li style="margin-bottom: 8px; line-height: 1.5;">{finding}</li>\n'
    
    if not key_findings:
        html += '            <li style="margin-bottom: 8px;">No confirmed facts extracted from sources.</li>\n'
    
    html += '''        </ul>
    </div>
    
    <div style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 700; font-size: 14px; background: #FEF2F2; color: #991B1B; margin-bottom: 15px;">
        ⚠ UNVERIFIED CLAIMS
    </div>
    <div style="background: #FEF2F2; border-left: 3px solid #EF4444; padding: 15px 20px; margin: 10px 0 25px 0; border-radius: 6px;">
        <ul style="margin: 0; padding-left: 20px; color: #991B1B; list-style-type: disc;">
'''
    for claim in unverified[:5]:
        html += f'            <li style="margin-bottom: 8px; line-height: 1.5;">{claim}</li>\n'
    
    if not unverified:
        html += '            <li style="margin-bottom: 8px;">No unverified claims detected.</li>\n'
    
    html += f'''        </ul>
    </div>
    
    <div style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 700; font-size: 14px; background: #F3F4F6; color: #374151; margin-bottom: 15px;">
        📋 EXECUTIVE SUMMARY
    </div>
    <div style="background: #F9FAFB; border-left: 3px solid #6B7280; padding: 15px 20px; margin: 10px 0 0 0; border-radius: 6px; color: #374151; line-height: 1.7;">
        {summary.strip() if summary.strip() else "Analysis complete. See findings above for details."}
    </div>
</div>
<script>
    // Send height to Streamlit for auto-sizing
    const height = document.body.scrollHeight;
    window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: height}}, '*');
</script>
'''
    return html, confidence

def wrap_html_for_autosize(html_content):
    """Wrap HTML content with auto-resize JavaScript."""
    return f'''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * {{ font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: transparent; }}
</style>
{html_content}
<script>
    // Auto-resize iframe to content height
    function sendHeight() {{
        const height = document.body.scrollHeight + 20;
        window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: height}}, '*');
    }}
    window.addEventListener('load', sendHeight);
    setTimeout(sendHeight, 100);
    setTimeout(sendHeight, 500);
</script>
'''

# ============================================
# CUSTOM CSS - GREY/TEAL BRANDING
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F5F7FA 0%, #E8ECF1 100%);
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 20px 20px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 42px;
        font-weight: 700;
        color: #374151;
        margin-bottom: 8px;
    }
    .hero-title span {
        color: #3D8B52;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #6B7280;
        margin-bottom: 20px;
    }
    
    /* Report Box */
    .report-box {
        background: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        border-left: 4px solid #3D8B52;
        color: #374151;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        line-height: 1.7;
    }
    .report-title {
        color: #3D8B52;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #E5E7EB;
    }
    .report-box strong, .report-box b {
        color: #1F2937;
        font-weight: 700;
    }
    .report-box .section-header {
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
        color: #3D8B52;
        font-size: 14px;
        letter-spacing: 0.5px;
        padding: 8px 12px;
        background: #F0FDF4;
        border-radius: 6px;
        display: inline-block;
    }
    .report-box .unverified {
        background: #FEF2F2;
        border-left: 3px solid #EF4444;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 6px;
        color: #991B1B;
    }
    .report-box .verified {
        background: #F0FDF4;
        border-left: 3px solid #3D8B52;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 6px;
        color: #166534;
    }
    .report-box ul {
        padding-left: 20px;
        color: #4B5563;
        margin: 10px 0;
    }
    .report-box li {
        margin-bottom: 8px;
        line-height: 1.6;
        padding-left: 5px;
    }
    
    /* Confidence Badge */
    .confidence-large {
        font-size: 48px;
        font-weight: 700;
        color: #3D8B52;
        text-align: center;
    }
    .confidence-label {
        font-size: 14px;
        color: #6B7280;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sources Card */
    .sources-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .sources-title {
        color: #374151;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        color: #374151;
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3D8B52;
        box-shadow: 0 0 0 2px rgba(0, 131, 143, 0.2);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3D8B52 0%, #4CAF50 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("<h3 style='margin:0;'><span style='color:#3D8B52;'>Counter</span><span style='color:#374151;'>Point</span></h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Inter, sans-serif; font-size: 14px; color: #6B7280; margin-top: -10px;'>News Verification AI</p>", unsafe_allow_html=True)
    st.divider()
    demo_mode = st.checkbox("🎯 Enable Demo Mode", value=False)
    st.divider()
    if st.button("⚙️ API Settings"):
        st.switch_page("pages/Settings.py")

# ============================================
# HERO SECTION (Logo removed per user request)
# ============================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title"><span style="color: #3D8B52;">Counter</span><span style="color: #374151;">Point</span></div>
    <div class="hero-subtitle">AI-Powered News Verification for Journalists</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# SEARCH FORM
# ============================================
with st.container():
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        with st.form(key="search_form"):
            topic = st.text_input("Search Topic", placeholder="Enter a news topic to verify...", label_visibility="collapsed")
            submit_button = st.form_submit_button("🔍 Analyze", use_container_width=True)

# ============================================
# HANDLE SEARCH
# ============================================
def display_result(result_data):
    """Display verification results with professional layout."""
    topic_name = result_data['topic']
    confidence = result_data['confidence']
    report_content = result_data.get('report_content', '')
    report_html = result_data.get('report_html', '')
    sources = result_data['sources']
    
    # Parse confidence for color coding
    try:
        conf_num = int(str(confidence).replace('%', '').replace('N/A', '0'))
    except:
        conf_num = 0
    
    if conf_num >= 70:
        conf_color = "#3D8B52"
        conf_label = "High Confidence"
    elif conf_num >= 40:
        conf_color = "#F59E0B"
        conf_label = "Medium Confidence"
    else:
        conf_color = "#EF4444"
        conf_label = "Low Confidence"
    
    # ===== HEADER ROW: Title + Confidence =====
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 28px; font-weight: 700; color: #374151;">Verification Report</div>
            <div style="font-size: 18px; color: #6B7280; margin-top: 5px;">{topic_name}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Confidence Score</div>
            <div style="font-size: 52px; font-weight: 700; color: {conf_color}; line-height: 1;">{confidence}%</div>
            <div style="font-size: 12px; color: {conf_color}; margin-top: 8px; font-weight: 600;">{conf_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== MAIN CONTENT: Report + Sources Sidebar =====
    main_col, sources_col = st.columns([2.5, 1])
    
    with main_col:
        # Use components.html for proper HTML rendering without escaping
        if report_html:
            components.html(report_html, height=700, scrolling=True)
        elif report_content:
            # Parse and format the Gemini response
            html_content = f"""
            <div style="background: #FFFFFF; padding: 30px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); font-family: 'Inter', sans-serif;">
                {report_content.replace(chr(10), '<br>')}
            </div>
            """
            components.html(html_content, height=700, scrolling=True)
    
    with sources_col:
        st.markdown("""
        <div style="font-size: 14px; font-weight: 700; color: #374151; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.5px;">
            📰 Sources
        </div>
        """, unsafe_allow_html=True)
        
        for i, source in enumerate(sources[:6]):  # Limit to 6 sources
            if isinstance(source, dict):
                title = source.get('title', 'Article')
                title_display = title[:40] + '...' if len(title) > 40 else title
                url = source.get('url', '#')
                # Extract domain
                try:
                    domain = url.split('/')[2].replace('www.', '')
                except:
                    domain = 'Source'
                
                st.markdown(f"""
                <a href="{url}" target="_blank" style="text-decoration: none; display: block;">
                    <div style="background: #FFFFFF; border-radius: 10px; padding: 14px; margin-bottom: 10px; border-left: 4px solid #3D8B52; box-shadow: 0 2px 6px rgba(0,0,0,0.04); transition: transform 0.2s, box-shadow 0.2s;">
                        <div style="font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 6px; line-height: 1.4;">{title_display}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 11px; color: #3D8B52; font-weight: 500;">{domain}</span>
                            <span style="font-size: 10px; color: #9CA3AF;">Recent</span>
                        </div>
                    </div>
                </a>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"• {source}")

def display_x_intel(x_data, topic):
    """Display X.com Intel data from Grok."""
    if not x_data:
        st.info("No X.com data available. Click 'Analyze' to fetch X intelligence.")
        return
    
    if "error" in x_data:
        st.error(f"⚠️ {x_data['error']}")
        if "raw" in x_data:
            st.code(x_data['raw'], language="text")
        return
    
    # X Summary
    x_summary = x_data.get('x_summary', 'No summary available.')
    st.info(f"**📊 X.com Summary:** {x_summary}")
    
    # Viral Rumors
    viral_rumors = x_data.get('viral_rumors', [])
    if viral_rumors:
        st.markdown("### ⚠️ Viral Rumors/Claims on X")
        for rumor in viral_rumors:
            st.warning(f"🔥 {rumor}")
    else:
        st.success("No viral rumors detected on X.")
    
    # Sources (Tweet Cards)
    sources = x_data.get('sources', [])
    if sources:
        st.markdown("### 🐦 Featured Posts")
        for source in sources[:5]:  # Limit to 5 tweets
            handle = source.get('handle', '@unknown')
            link = source.get('link', '#')
            text = source.get('text', 'No text available')
            
            st.markdown(f"""
            <a href="{link}" target="_blank" style="text-decoration: none; display: block;">
                <div style="background: #FFFFFF; border-radius: 12px; padding: 16px; margin-bottom: 12px; border-left: 4px solid #1DA1F2; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                    <div style="font-size: 14px; font-weight: 600; color: #1DA1F2; margin-bottom: 8px;">{handle}</div>
                    <div style="font-size: 13px; color: #374151; line-height: 1.5;">{text}</div>
                    <div style="font-size: 11px; color: #9CA3AF; margin-top: 8px;">View on X →</div>
                </div>
            </a>
            """, unsafe_allow_html=True)
    else:
        st.info("No featured posts found.")

def display_results_stacked(result_data, x_intel_data, topic):
    """Display results in stacked layout with Global Web News above X.com Intel."""
    
    # ===== SECTION 1: Global Web News =====
    st.markdown("""
    <div style="font-size: 20px; font-weight: 700; color: #374151; margin: 20px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #3D8B52;">
        🌐 Global Web News
    </div>
    """, unsafe_allow_html=True)
    
    if result_data:
        display_result(result_data)
    else:
        st.info("No web news data available.")
    
    # ===== SECTION 2: X.com Intel =====
    st.markdown("""
    <div style="font-size: 20px; font-weight: 700; color: #374151; margin: 40px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #1DA1F2;">
        🐦 X.com Intel
    </div>
    """, unsafe_allow_html=True)
    
    display_x_intel(x_intel_data, topic)


if submit_button:
    tavily_key = os.getenv("TAVILY_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not topic:
        st.warning("Please enter a topic to verify.")
    
    # DEMO MODE
    elif demo_mode and topic.strip().lower() == "dubai storm":
        with st.status("Analyzing...", expanded=True) as status:
            st.write("🔍 Searching global sources...")
            st.write("✅ Found 5 articles")
            st.write("🤖 Running AI fact-check...")
            st.write("✅ Complete")
            status.update(label="✅ Verification Complete", state="complete", expanded=False)
        
        report_html = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
    body { background: transparent; }
</style>
<div style="background: #FFFFFF; padding: 30px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); font-family: 'Inter', sans-serif;">
    
    <div style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 700; font-size: 14px; background: #F0FDF4; color: #166534; margin-bottom: 15px;">
        ✓ CONFIRMED FACTS
    </div>
    <div style="background: #F0FDF4; border-left: 3px solid #3D8B52; padding: 15px 20px; margin: 10px 0 25px 0; border-radius: 6px;">
        <ul style="margin: 0; padding-left: 20px; color: #166534; list-style-type: disc;">
            <li style="margin-bottom: 8px; line-height: 1.5;">Heavy rainfall and thunderstorms have struck Dubai and other parts of the UAE.</li>
            <li style="margin-bottom: 8px; line-height: 1.5;">Flights at Dubai International Airport (DXB) have been disrupted with cancellations and diversions.</li>
            <li style="margin-bottom: 8px; line-height: 1.5;">Schools and government offices have switched to remote work due to weather conditions.</li>
            <li style="margin-bottom: 0; line-height: 1.5;">Authorities have issued weather warnings advising residents to stay indoors.</li>
        </ul>
    </div>
    
    <div style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 700; font-size: 14px; background: #FEF2F2; color: #991B1B; margin-bottom: 15px;">
        ⚠ UNVERIFIED RUMORS
    </div>
    <div style="background: #FEF2F2; border-left: 3px solid #EF4444; padding: 15px 20px; margin: 10px 0 25px 0; border-radius: 6px;">
        <ul style="margin: 0; padding-left: 20px; color: #991B1B; list-style-type: disc;">
            <li style="margin-bottom: 8px; line-height: 1.5;"><strong>Rumor:</strong> The Burj Khalifa has been struck by lightning and structurally damaged. → <span style="color:#EF4444; font-weight:bold;">FALSE</span></li>
            <li style="margin-bottom: 0; line-height: 1.5;"><strong>Rumor:</strong> All roads in Dubai are completely closed. → <span style="color:#EF4444; font-weight:bold;">FALSE</span></li>
        </ul>
    </div>
    
    <div style="display: inline-block; padding: 8px 16px; border-radius: 6px; font-weight: 700; font-size: 14px; background: #F3F4F6; color: #374151; margin-bottom: 15px;">
        📋 EXECUTIVE SUMMARY
    </div>
    <div style="background: #F9FAFB; border-left: 3px solid #6B7280; padding: 15px 20px; margin: 10px 0 0 0; border-radius: 6px; color: #374151; line-height: 1.7;">
        The event is a significant weather anomaly causing widespread disruption across the UAE. Official government and news sources confirm the severity of the rainfall and its impact on transportation and daily activities. However, claims of major structural damage to landmarks like Burj Khalifa have been debunked by authorities.
    </div>
</div>
<script>
    function sendHeight() {
        const height = document.body.scrollHeight + 20;
        window.parent.postMessage({type: 'streamlit:setFrameHeight', height: height}, '*');
    }
    window.addEventListener('load', sendHeight);
    setTimeout(sendHeight, 100);
    setTimeout(sendHeight, 500);
</script>
"""
        sources = [
            {"title": "Gulf News Live Updates", "url": "https://gulfnews.com"},
            {"title": "Khaleej Times Weather Report", "url": "https://khaleejtimes.com"},
            {"title": "UAE National Centre of Meteorology", "url": "https://ncm.ae"},
            {"title": "Emirates 24/7 News", "url": "https://emirates247.com"},
            {"title": "The National UAE", "url": "https://thenationalnews.com"}
        ]
        
        # Save to history (will save x_intel after it's set)
        x_intel_demo = {
            "x_summary": "Dubai Storm is trending with residents sharing dramatic flooding videos. Mixed reactions between concern for safety and amazement at unusual weather.",
            "viral_rumors": [
                "Claims that Burj Khalifa was struck by lightning - UNVERIFIED",
                "Reports of 'cloud seeding' causing the storm - DEBATED"
            ],
            "sources": [
                {"handle": "@DubaiMediaOffice", "link": "https://x.com/DubaiMediaOffice", "text": "Heavy rainfall expected. Residents advised to stay indoors and avoid unnecessary travel."},
                {"handle": "@weatheruae", "link": "https://x.com/weatheruae", "text": "Historic rainfall levels recorded in Dubai. Stay safe everyone!"},
                {"handle": "@gaborsteingart", "link": "https://x.com/gaborsteingart", "text": "Incredible scenes from Dubai airport as floods disrupt flights."}
            ]
        }
        save_search("Dubai Storm (Demo)", 100, report_html, x_intel_demo)
        
        # Store in session state
        st.session_state.last_result = {
            'topic': "Dubai Storm",
            'confidence': 100,
            'report_html': report_html,
            'sources': sources
        }
        
        # Fetch X.com Intel (demo: use mock data)
        st.session_state.x_intel_result = {
            "x_summary": "Dubai Storm is trending with residents sharing dramatic flooding videos. Mixed reactions between concern for safety and amazement at unusual weather.",
            "viral_rumors": [
                "Claims that Burj Khalifa was struck by lightning - UNVERIFIED",
                "Reports of 'cloud seeding' causing the storm - DEBATED"
            ],
            "sources": [
                {"handle": "@DubaiMediaOffice", "link": "https://x.com/DubaiMediaOffice", "text": "Heavy rainfall expected. Residents advised to stay indoors and avoid unnecessary travel."},
                {"handle": "@weatheruae", "link": "https://x.com/weatheruae", "text": "Historic rainfall levels recorded in Dubai. Stay safe everyone!"},
                {"handle": "@gaborsteingart", "link": "https://x.com/gaborsteingart", "text": "Incredible scenes from Dubai airport as floods disrupt flights."}
            ]
        }
        
        display_results_stacked(st.session_state.last_result, st.session_state.x_intel_result, "Dubai Storm")
    
    # REAL MODE
    elif not tavily_key or not gemini_key:
        st.error("Missing API Keys. Please configure them in **Settings**.")
    else:
        news_results = None
        verification_report = None
        
        with st.status("Analyzing...", expanded=True) as status:
            st.write("🔍 Searching global sources...")
            news_results = get_live_news(topic)
            
            if news_results:
                st.write(f"✅ Found {len(news_results)} articles")
                st.write("🤖 Running AI fact-check...")
                verification_report = verify_news(topic, news_results)
                st.write("🐦 Fetching X.com Intel...")
                x_intel = get_x_intel(topic)
                st.write("✅ Complete")
                status.update(label="✅ Verification Complete", state="complete", expanded=False)
            else:
                status.update(label="❌ Search Failed", state="error")
        
        if news_results and verification_report:
            # Parse Gemini response into styled HTML
            report_html, confidence_score = parse_gemini_to_html(verification_report, topic)
            
            # Save to history with X intel data
            save_search(topic, confidence_score, report_html, x_intel)
            
            # Store in session state
            st.session_state.last_result = {
                'topic': topic,
                'confidence': confidence_score,
                'report_html': report_html,
                'sources': news_results
            }
            st.session_state.x_intel_result = x_intel
            
            display_results_stacked(st.session_state.last_result, st.session_state.x_intel_result, topic)

# ============================================
# DISPLAY PREVIOUS RESULT (if exists and no new search)
# ============================================
elif st.session_state.last_result:
    st.markdown("---")
    st.markdown("### 📋 Previous Result")
    display_results_stacked(
        st.session_state.last_result, 
        st.session_state.x_intel_result, 
        st.session_state.last_result.get('topic', 'Unknown')
    )
