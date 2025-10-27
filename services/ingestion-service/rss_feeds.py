FEEDS_TO_SCRAPE = {
    "feeds": {
        # --- 1. AI & Machine Learning ---
        "ai_ml": [
            # --- Foundational Research & Big Labs ---
            {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml"},
            {"name": "Microsoft AI Blog", "url": "https://blogs.microsoft.com/ai/feed/"},
            {"name": "DeepMind Blog", "url": "https://deepmind.google/blog/rss.xml"},
            {"name": "Google AI Blog", "url": "https://blog.google/technology/ai/rss/"},
            {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml"},
            # --- Academic & Papers ---
            {"name": "arXiv AI (cs.AI)", "url": "https://arxiv.org/rss/cs.AI"},
            {"name": "arXiv Machine Learning (cs.LG)", "url": "https://arxiv.org/rss/cs.LG"},
            {"name": "arXiv Computation & Language (cs.CL)", "url": "https://arxiv.org/rss/cs.CL"},
            {"name": "arXiv Computer Vision (cs.CV)", "url": "https://arxiv.org/rss/cs.CV"},
            # --- AI News, Analysis & Niche ---
            {"name": "KDnuggets News", "url": "https://www.kdnuggets.com/feed"},
            {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
            {"name": "The Gradient", "url": "https://thegradient.pub/rss/"},
            {"name": "AI Alignment Forum", "url": "https://www.alignmentforum.org/feed.xml?view=rss"},
            {"name": "Analytics Vidhya", "url": "https://www.analyticsvidhya.com/feed/"},
        ],
        
        # --- 2. General Technology ---
        "general_tech": [
            # --- Major Tech News ---
            {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
            {"name": "Wired", "url": "https://www.wired.com/feed/rss"},
            {"name": "Ars Technica", "url": "https://arstechnica.com/feed/"},
            {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
            {"name": "Techmeme", "url": "https://www.techmeme.com/feed.xml"},
            {"name": "Engadget", "url": "https://www.engadget.com/rss.xml"},
            {"name": "Gizmodo", "url": "https://gizmodo.com/rss"},
            {"name": "ZDNet", "url": "https://www.zdnet.com/news/rss.xml"},
            {"name": "Tech in Asia", "url": "https://www.techinasia.com/feed"},
            # --- Developer & Community ---
            {"name": "Hacker News (Top Stories)", "url": "https://news.ycombinator.com/rss"},
            {"name": "HackerNoon", "url": "https://hackernoon.com/feed"},
            {"name": "Slashdot", "url": "http://rss.slashdot.org/Slashdot/slashdotMain"},
            {"name": "Lobsters", "url": "https://lobste.rs/rss"},
        ],

        # --- 3. Finance & Business ---
        "finance_business": [
            # --- Global Finance & Markets ---
            {"name": "Bloomberg Technology", "url": "https://feeds.bloomberg.com/technology/news.rss"},
            {"name": "Bloomberg Business", "url": "https://feeds.bloomberg.com/business/news.rss"},
            {"name": "Financial Times World News", "url": "https://www.ft.com/world?format=rss"}, # Paywall
            {"name": "Wall Street Journal World News", "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml"}, # Paywall
            {"name": "Wall Street Journal Business", "url": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"}, # Paywall
            {"name": "CNBC Top News", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
            {"name": "MarketWatch Top Stories", "url": "http://feeds.marketwatch.com/marketwatch/topstories/"},
            {"name": "The Economist (Business & Finance)", "url": "https://www.economist.com/finance-and-economics/rss.xml"}, # Paywall
            {"name": "Quartz", "url": "https://cms.qz.com/feed/"},
            {"name": "Forbes Business", "url": "https://www.forbes.com/business/feed/"},
            # --- Crypto & Digital Assets ---
            {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
            {"name": "The Block", "url": "https://www.theblock.co/rss.xml"},
        ],

        # --- 4. Economics (Macro) ---
        "economics_macro": [
            {"name": "St. Louis Fed (FRED) Blog", "url": "https://fredblog.stlouisfed.org/feed/"},
            {"name": "IMF News", "url": "https://www.imf.org/en/News/rss?type=News-Articles"},
        ],

        # --- 5. Politics & Government ---
        "politics_government": [
            {"name": "The Hill", "url": "https://thehill.com/news/feed/"},
            {"name": "Axios Politics", "url": "https://api.axios.com/feed/politics"},
            {"name": "Foreign Policy", "url": "https://foreignpolicy.com/feed/"},
            {"name": "Politico", "url": "https://www.politico.com/rss/politicopicks.xml"},
        ],

        # --- 6. World News ---
        "world_news": [
            {"name": "BBC News World", "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
            {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml"},
            {"name": "The Economist World", "url": "https://www.economist.com/the-world-this-week/rss.xml"},
            {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss"},
            {"name": "Al Jazeera English", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
            {"name": "South China Morning Post (SCMP)", "url": "https://www.scmp.com/rss/91/feed"},
            {"name": "Der Spiegel International", "url": "https://www.spiegel.de/international/index.rss"},
            {"name": "Axios World", "url": "https://api.axios.com/feed/world"},
        ],

        # --- 7. India Specific News ---
        "india_specific_news": [
            {"name": "The Hindu (National)", "url": "https://www.thehindu.com/news/national/feeder/default.rss"},
            {"name": "The Times of India (National)", "url": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"},
            {"name": "The Indian Express", "url": "https://indianexpress.com/feed/"},
            {"name": "Hindustan Times (National)", "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"},
            {"name": "Livemint (Business)", "url": "https://www.livemint.com/rss/companies"},
        ],

        # --- 8. Cybersecurity ---
        "cybersecurity": [
            {"name": "Krebs on Security", "url": "https://krebsonsecurity.com/feed/"},
            {"name": "Schneier on Security", "url": "https://www.schneier.com/feed/atom/"},
            {"name": "Dark Reading", "url": "https://www.darkreading.com/rss.xml"},
            {"name": "Bleeping Computer", "url": "https://www.bleepingcomputer.com/feed/"},
            {"name": "Wired Security", "url": "https://www.wired.com/feed/category/security/latest/rss"},
        ],

        # --- 9. Science & Environment ---
        "science_environment": [
            {"name": "Nature", "url": "https://www.nature.com/nature.rss"},
            {"name": "Science Magazine", "url": "https://www.science.org/rss/news_current.xml"},
            {"name": "New Scientist", "url": "https://www.newscientist.com/feed/home/"},
            {"name": "Ars Technica (Science)", "url": "https://arstechnica.com/science/feed/"},
            {"name": "The Guardian (Environment)", "url": "https://www.theguardian.com/environment/rss"},
            {"name": "NYT Climate", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Climate.xml"},
            {"name": "NASA News", "url": "httpsfs://www.nasa.gov/rss/dyn/breaking_news.rss"},
        ],

        # --- 10. Health & Medicine ---
        "health_medicine": [
            {"name": "STAT News", "url": "https://www.statnews.com/feed/"},
            {"name": "NPR Health", "url": "https://feeds.npr.org/1007/rss.xml"},
            {"name": "Fierce Pharma", "url": "https://www.fiercepharma.com/rss/xml"},
            {"name": "Fierce Biotech", "url": "https://www.fiercebiotech.com/rss/xml"},
        ],

        # --- 11. Sports ---
        "sports": [
            # --- Global Sports ---
            {"name": "ESPN Top Headlines", "url": "https://www.espn.com/espn/rss/news"},
            {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport/rss.xml"},
            {"name": "Yahoo Sports", "url": "https://sports.yahoo.com/rss/"},
            {"name": "Formula 1", "url": "https://www.formula1.com/content/fom-website/en/latest/all.xml"},
            # --- India Specific Sports ---
            {"name": "The Hindu Sports News", "url": "https://www.thehindu.com/sport/feeder/default.rss"},
            {"name": "ESPN Cricinfo", "url": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml"},
            {"name": "Times of India Sports", "url": "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms"},
        ],
        
        # --- 12. Culture & Entertainment ---
        "culture_entertainment": [
            {"name": "Variety", "url": "httpsServices://variety.com/feed/"},
            {"name": "The Hollywood Reporter", "url": "https://www.hollywoodreporter.com/feed/"},
            {"name": "Rolling Stone", "url": "https://www.rollingstone.com/feed/"},
            {"name": "Pitchfork", "url": "https://pitchfork.com/feed/rss"},
            {"name": "Kotaku (Gaming)", "url": "https://kotaku.com/rss"},
            {"name": "The Art Newspaper", "url": "https://www.theartnewspaper.com/rss.xml"},
        ]
    }
}