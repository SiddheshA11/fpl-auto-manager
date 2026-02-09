#!/usr/bin/env python3
"""
FPL Token Refresh via Headless Browser

Uses Playwright to automate FPL login and extract a fresh access token,
then updates the FPL_COOKIE GitHub secret for seamless automation.
"""

import os
import sys
import json
import base64
import logging
import requests
from nacl import encoding, public

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('token_refresh')


def encrypt_secret(public_key: str, secret_value: str) -> str:
    """Encrypt a secret for GitHub Actions using libsodium."""
    public_key_bytes = base64.b64decode(public_key)
    sealed_box = public.SealedBox(public.PublicKey(public_key_bytes))
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def update_github_secret(secret_name: str, secret_value: str) -> bool:
    """Update a GitHub Actions secret via the API."""
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT")
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    
    if not github_token or not github_repo:
        logger.error("GITHUB_TOKEN/GH_PAT or GITHUB_REPOSITORY not set")
        return False
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    # Get the public key for encryption
    key_url = f"https://api.github.com/repos/{github_repo}/actions/secrets/public-key"
    key_resp = requests.get(key_url, headers=headers)
    
    if key_resp.status_code != 200:
        logger.error(f"Failed to get public key: {key_resp.status_code}")
        return False
    
    key_data = key_resp.json()
    encrypted_value = encrypt_secret(key_data["key"], secret_value)
    
    # Update the secret
    secret_url = f"https://api.github.com/repos/{github_repo}/actions/secrets/{secret_name}"
    secret_resp = requests.put(
        secret_url,
        headers=headers,
        json={
            "encrypted_value": encrypted_value,
            "key_id": key_data["key_id"]
        }
    )
    
    if secret_resp.status_code in (201, 204):
        logger.info(f"Successfully updated {secret_name} secret")
        return True
    else:
        logger.error(f"Failed to update secret: {secret_resp.status_code} - {secret_resp.text}")
        return False


def get_token_via_browser() -> str | None:
    """Use Playwright to log into FPL and extract access token."""
    from playwright.sync_api import sync_playwright
    
    email = os.environ.get("FPL_EMAIL")
    password = os.environ.get("FPL_PASSWORD")
    
    if not email or not password:
        logger.error("FPL_EMAIL and FPL_PASSWORD must be set")
        return None
    
    logger.info("Starting headless browser...")
    
    with sync_playwright() as p:
        # Launch headless Chromium
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Navigate to FPL - this will redirect to login
            logger.info("Navigating to FPL...")
            page.goto("https://fantasy.premierleague.com/")
            
            # Wait for login button and click it
            logger.info("Looking for login button...")
            page.wait_for_timeout(2000)
            
            # Try to find and click login button
            try:
                login_btn = page.locator("text=Log in").first
                if login_btn.is_visible():
                    login_btn.click()
                    page.wait_for_timeout(2000)
            except:
                pass  # May already be on login page
            
            # Wait for email input and fill credentials
            logger.info("Entering credentials...")
            page.wait_for_selector('input[type="email"], input[name="email"], #loginEmail', timeout=15000)
            
            # Try different selectors for email field
            email_selectors = ['input[type="email"]', 'input[name="email"]', '#loginEmail', 'input[id*="email"]']
            for selector in email_selectors:
                try:
                    email_field = page.locator(selector).first
                    if email_field.is_visible():
                        email_field.fill(email)
                        break
                except:
                    continue
            
            # Try different selectors for password field
            password_selectors = ['input[type="password"]', 'input[name="password"]', '#loginPassword', 'input[id*="password"]']
            for selector in password_selectors:
                try:
                    pwd_field = page.locator(selector).first
                    if pwd_field.is_visible():
                        pwd_field.fill(password)
                        break
                except:
                    continue
            
            # Click sign in button
            logger.info("Clicking sign in...")
            submit_selectors = ['button[type="submit"]', 'button:has-text("Sign in")', 'input[type="submit"]', 'button:has-text("Log in")']
            for selector in submit_selectors:
                try:
                    submit_btn = page.locator(selector).first
                    if submit_btn.is_visible():
                        submit_btn.click()
                        break
                except:
                    continue
            
            # Wait for redirect back to FPL
            logger.info("Waiting for login to complete...")
            page.wait_for_timeout(5000)
            
            # Navigate to FPL to ensure we're logged in
            page.goto("https://fantasy.premierleague.com/")
            page.wait_for_timeout(3000)
            
            # Extract access token from localStorage
            logger.info("Extracting access token from localStorage...")
            access_token = page.evaluate("""
                () => {
                    const key = Object.keys(localStorage).find(k => k.startsWith('oidc.user:'));
                    if (key) {
                        const user = JSON.parse(localStorage.getItem(key));
                        return user.access_token;
                    }
                    return null;
                }
            """)
            
            if access_token:
                logger.info("Successfully extracted access token!")
                return access_token
            else:
                logger.error("Could not find access token in localStorage")
                # Take a screenshot for debugging
                page.screenshot(path="debug_screenshot.png")
                logger.info("Saved debug screenshot to debug_screenshot.png")
                return None
                
        except Exception as e:
            logger.error(f"Browser automation error: {e}")
            try:
                page.screenshot(path="error_screenshot.png")
                logger.info("Saved error screenshot")
            except:
                pass
            return None
        finally:
            browser.close()


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("FPL Token Refresh - Automated Browser Login")
    logger.info("=" * 60)
    
    # Get fresh token via browser
    access_token = get_token_via_browser()
    
    if not access_token:
        logger.error("Failed to obtain access token")
        sys.exit(1)
    
    # Format as cookie string
    cookie_value = f"access_token={access_token}"
    
    # Update GitHub secret
    if update_github_secret("FPL_COOKIE", cookie_value):
        logger.info("Token refresh complete! FPL_COOKIE updated successfully.")
        sys.exit(0)
    else:
        logger.error("Failed to update FPL_COOKIE secret")
        sys.exit(1)


if __name__ == "__main__":
    main()
