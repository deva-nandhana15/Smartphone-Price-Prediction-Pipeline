import requests
import json
import time
import logging
from datetime import datetime
from typing import List, Dict
import sys
sys.path.append('/opt/config')
from ebay_config import EBAY_CONFIG, HDFS_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EbayDataIngestion:
    def __init__(self):
        self.app_id = EBAY_CONFIG['app_id']
        self.client_secret = EBAY_CONFIG['client_secret']
        self.base_url = 'https://api.ebay.com/buy/browse/v1'
        self.access_token = None
        self.category_id = EBAY_CONFIG['category_id']
        self.search_keywords = EBAY_CONFIG['search_keywords']
        self.items_per_page = EBAY_CONFIG['items_per_page']
        self.max_pages = EBAY_CONFIG['max_pages']
        self.rate_limit_delay = EBAY_CONFIG['rate_limit_delay']
        self.max_retries = EBAY_CONFIG['max_retries']
        
    def get_access_token(self) -> str:
        """Obtain OAuth access token from eBay API"""
        auth_url = 'https://api.ebay.com/identity/v1/oauth2/token'
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'scope': 'https://api.ebay.com/oauth/api_scope'
        }
        
        try:
            response = requests.post(
                auth_url,
                headers=headers,
                data=data,
                auth=(self.app_id, self.client_secret)
            )
            response.raise_for_status()
            self.access_token = response.json()['access_token']
            logger.info("Successfully obtained access token")
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to obtain access token: {e}")
            raise
    
    def search_items(self, keyword: str, offset: int = 0) -> Dict:
        """Search for items using eBay Browse API"""
        if not self.access_token:
            self.get_access_token()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
        }
        
        params = {
            'q': keyword,
            'category_ids': self.category_id,
            'limit': self.items_per_page,
            'offset': offset,
            'filter': 'conditionIds:{1000|1500|2000|2500|3000}'  # New to Acceptable
        }
        
        url = f"{self.base_url}/item_summary/search"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch data after {self.max_retries} attempts")
                    return {}
    
    def extract_item_details(self, item: Dict) -> Dict:
        """Extract relevant fields from API response"""
        try:
            price_value = None
            if 'price' in item:
                price_value = float(item['price'].get('value', 0))
            elif 'currentBidPrice' in item:
                price_value = float(item['currentBidPrice'].get('value', 0))
            
            return {
                'item_id': item.get('itemId'),
                'title': item.get('title'),
                'price': price_value,
                'currency': item.get('price', {}).get('currency', 'USD'),
                'condition': item.get('condition'),
                'condition_id': item.get('conditionId'),
                'category_path': item.get('categoryPath'),
                'item_location': item.get('itemLocation', {}).get('country'),
                'seller_username': item.get('seller', {}).get('username'),
                'seller_feedback_score': item.get('seller', {}).get('feedbackScore'),
                'image_url': item.get('image', {}).get('imageUrl'),
                'item_web_url': item.get('itemWebUrl'),
                'shipping_options': item.get('shippingOptions'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting item details: {e}")
            return {}
    
    def ingest_data(self) -> List[Dict]:
        """Main ingestion function"""
        all_items = []
        total_api_calls = 0
        
        for keyword in self.search_keywords:
            logger.info(f"Searching for keyword: {keyword}")
            
            for page in range(self.max_pages):
                offset = page * self.items_per_page
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                response = self.search_items(keyword, offset)
                total_api_calls += 1
                
                if not response or 'itemSummaries' not in response:
                    logger.warning(f"No items found for {keyword} at offset {offset}")
                    break
                
                items = response.get('itemSummaries', [])
                if not items:
                    break
                
                for item in items:
                    item_details = self.extract_item_details(item)
                    if item_details and item_details.get('price'):
                        all_items.append(item_details)
                
                logger.info(f"Processed page {page + 1} for {keyword}. "
                          f"Total items so far: {len(all_items)}")
                
                # Stop if we've reached the total available
                if len(items) < self.items_per_page:
                    break
                
                # Check daily API limit
                if total_api_calls >= 4900:  # Conservative limit
                    logger.warning("Approaching daily API limit. Stopping ingestion.")
                    break
            
            if total_api_calls >= 4900:
                break
        
        logger.info(f"Total items collected: {len(all_items)}")
        return all_items
    
    def save_to_local(self, data: List[Dict], output_path: str):
        """Save data to local JSON file (for testing)"""
        import os
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_path}/ebay_data_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data saved to {filename}")
        return filename
    
    def save_to_hdfs(self, data: List[Dict]):
        """Save data to HDFS"""
        from hdfs import InsecureClient
        
        hdfs_client = InsecureClient(
            f"http://{HDFS_CONFIG['namenode_host']}:9870"
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hdfs_path = f"{HDFS_CONFIG['raw_data_path']}/ebay_data_{timestamp}.json"
        
        # Create directory if it doesn't exist
        try:
            hdfs_client.makedirs(HDFS_CONFIG['raw_data_path'])
        except:
            pass
        
        # Write data
        json_data = json.dumps(data, indent=2)
        hdfs_client.write(hdfs_path, json_data, encoding='utf-8')
        
        logger.info(f"Data saved to HDFS: {hdfs_path}")
        return hdfs_path


def main():
    ingestion = EbayDataIngestion()
    
    logger.info("Starting data ingestion...")
    data = ingestion.ingest_data()
    
    if data:
        # Save to local for backup
        ingestion.save_to_local(data, '/opt/pipeline/data')
        
        # Save to HDFS
        try:
            ingestion.save_to_hdfs(data)
        except Exception as e:
            logger.error(f"Failed to save to HDFS: {e}")
            logger.info("Data saved locally as backup")
    else:
        logger.error("No data collected")


if __name__ == '__main__':
    main()
