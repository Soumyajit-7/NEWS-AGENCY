import json
import os

import requests
from langchain.tools import tool

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from Utils import print_agent_output
from Agents import NewsAgents

from Utils import GROQ_LLM
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# search_tool = DuckDuckGoSearchRun()
search_tool = SerperDevTool()

news_scraper_agent = NewsAgents().make_news_scraper_agent()
event_detection_agent = NewsAgents().make_event_detection_agent()
fact_checking_agent = NewsAgents().make_fact_checking_agent()
content_generation_agent = NewsAgents().make_content_generation_agent()
editor_agent = NewsAgents().make_editor_agent()
Search_scraper_agent = NewsAgents().search_news_scraper_agent()
Search_fact_checker = NewsAgents().Search_make_fact_checking_agent()


class NewsAgencyTasks():
    def scrape_news(self):
        return Task(
            description=f"""Collect news data from various sources like news websites, social media, and official press releases. 
            Gather the latest news articles, tweets, and other relevant information. 
            Ensure that the data collected is relevant, up-to-date, and covers a broad range of topics.

            Output should include the source, timestamp, headline, and a brief summary of each news item.""",
            expected_output="""A JSON file containing a list of news items. Each item should have:
            - source (e.g., 'BBC', 'Twitter')
            - timestamp (e.g., '2024-06-29T12:34:56Z')
            - headline (e.g., 'Breaking: Major Event Happens')
            - summary (e.g., 'A brief summary of the event.')

            Example:
            [
                {
                    "source": "BBC",
                    "timestamp": "2024-06-29T12:34:56Z",
                    "headline": "Breaking: Major Event Happens",
                    "summary": "A brief summary of the event."
                },
                ...
            ]""",
            output_file=f"scraped_news.json",
            agent=news_scraper_agent
        )

    def headline_scrape_news(self, headline):
        return Task(
            description=f"""News Headline:\n\n {headline} \n\n
            Collect news data on the headline topic mentioned from various sources like news websites, social media, and official press releases.
            Gather the latest news articles, tweets, and other relevant information. 
            Ensure that the data collected is relevant, up-to-date, and covers a broad range of topics.
            
            Output should include the source, timestamp, headline, and a the entire news of each news item.""",
            expected_output="""A Detailed informaton about the scraped information from the web in string in .txt""",
            output_file=f"scraped_news.txt",
            agent=Search_scraper_agent
        )

    def detect_events(self):
        return Task(
            description=f"""Analyze the collected news data to identify significant news events. 
            Use NLP techniques to process and analyze the data, clustering similar news items to identify major events.
            Focus on identifying clusters that represent significant, trending, or breaking news.

            Ensure that each detected event has a clear description, a list of related news items, and any relevant metadata.""",
            expected_output="""A JSON file containing a list of detected events. Each event should include:
            - event_id (e.g., 'event_001')
            - description (e.g., 'Major Event Detected')
            - related_news (a list of news item IDs or summaries)
            - metadata (any additional relevant information)

            Example:
            [
                {
                    "event_id": "event_001",
                    "description": "Major Event Detected",
                    "related_news": ["news_001", "news_002", ...],
                    "metadata": {"location": "Global", "impact": "High"}
                },
                ...
            ]""",
            context=[self.scrape_news()],
            output_file=f"detected_events.json",
            agent=event_detection_agent
        )

    def fact_check_events(self):
        return Task(
            description=f"""Verify the accuracy of the detected news events by cross-referencing with reliable sources. 
            Search reputable databases, official statements, and trusted news sources to confirm the facts.
            Ensure that each fact-checked event includes citations of the sources used for verification.

            Focus on identifying any discrepancies and confirming the key facts of each event.""",
            expected_output="""A JSON file containing the fact-checked events. Each event should include:
            - event_id (e.g., 'event_001')
            - verified (True/False)
            - facts (a list of verified facts)
            - sources (a list of sources used for verification)

            Example:
            [
                {
                    "event_id": "event_001",
                    "verified": True,
                    "facts": ["Fact 1", "Fact 2", ...],
                    "sources": ["Source 1", "Source 2", ...]
                },
                ...
            ]""",

            context=[self.detect_events()],
            output_file=f"fact_checked_events.txt",
            agent=fact_checking_agent
        )

    def S_fact_check_events(self, headline):
        return Task(
            description=f"""Verify the accuracy of the detected news News Headline: {headline} events by cross-referencing with reliable sources. 
            Search reputable databases, official statements, and trusted news sources to confirm the facts.
            Ensure that each fact-checked event includes citations of the sources used for verification.

            Focus on identifying any discrepancies and confirming the key facts of each event.""",
            expected_output=""" A comprehensive detailed information after checking the facts.
            """,
            context=[self.headline_scrape_news(headline)],
            output_file=f"fact_checked_events.txt",
            agent=Search_fact_checker

        )

    def generate_news_content(self, headline):
        return Task(
            description=f"""Write comprehensive news articles based on the verified information on {headline} . on the Context.
            Generate coherent and engaging news articles, ensuring the tone and style are appropriate for the target audience.
            Each article should be clear, concise, and provide a thorough overview of the event.
            Read the responses very carefully.

            Include headlines, subheadings, and paragraphs. Ensure that the articles are structured logically and are easy to read.""",
            expected_output="""A JSON file containing the generated news articles. Make the contents highly detailed.

            For Example:
            [
                {
                    "headline": "headline",
                    "subheadings": ["Introduction", "Details", "Conclusion"],
                    "content": "<p>Introduction...</p><p>Details...</p><p>Conclusion...</p>"
                },
                ...
            ]""",
            context=[self.S_fact_check_events(headline)],
            output_file=f"news_articles.json",
            agent=content_generation_agent
        )

    def edit_news_content(self, headline):
        return Task(
            description=f"""Review and edit the generated articles for quality and consistency.
            Check for grammar, clarity, coherence, and adherence to editorial guidelines.
            Make sure that each article is well-written, accurate, and engaging.

            Provide feedback and make necessary revisions to ensure the highest quality.""",
            expected_output="""A JSON file containing the edited news articles. Each article should include:
            - headline (e.g., 'Major Event Happens')
            - subheadings (e.g., ['Introduction', 'Details', 'Conclusion'])
            - content (a structured text with paragraphs, edited for quality)

            For Example:
            [
                {
                    "headline": "Major Event Happens",
                    "subheadings": ["Introduction", "Details", "Conclusion"],
                    "content": "<p>Introduction...</p><p>Details...</p><p>Conclusion...</p>"
                },
                ...
            ]
            
            generate only one response.""",
            context=[self.generate_news_content(headline)],
            # output_file=f"edited_news_articles.json",
            agent=editor_agent
        )
