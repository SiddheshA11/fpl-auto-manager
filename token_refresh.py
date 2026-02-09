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
    github_token = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
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
            
            # Wait for page to load
            page.wait_for_timeout(3000)
            
            # Handle cookie consent dialog if present
            logger.info("Checking for cookie consent dialog...")
            try:
                cookie_btn = page.locator('button:has-text("Accept All Cookies")').first
                if cookie_btn.is_visible(timeout=3000):
                    logger.info("Accepting cookies...")
                    cookie_btn.click()
                    page.wait_for_timeout(1000)
            except:
                logger.info("No cookie dialog found, continuing...")
            
            # Try to find and click login button
            logger.info("Looking for login button...")
            try:
                login_btn = page.locator("text=Log in").first
                if login_btn.is_visible(timeout=3000):
                    login_btn.click()
                    page.wait_for_timeout(3000)
            except:
                logger.info("No login button found, may already be on auth page...")
            
            # Wait for login page to load
            logger.info("Waiting for login page...")
            page.wait_for_timeout(2000)
            
            # Take screenshot to debug what page we're on
            page.screenshot(path="login_page.png")
            logger.info("Saved login page screenshot")
            
            # Fill in email - use placeholder matching from actual FPL login page
            logger.info("Entering credentials...")
            
            # The FPL login page has inputs with placeholders "Email address" and "Password"
            email_field = page.locator('input[placeholder="Email address"]').first
            try:
                email_field.wait_for(state="visible", timeout=15000)
                email_field.fill(email)
                logger.info("Filled email field")
            except:
                # Fallback: try all visible text inputs
                logger.warning("Placeholder selector failed, trying fallback...")
                inputs = page.locator('input:visible').all()
                logger.info(f"Found {len(inputs)} visible inputs")
                for i, inp in enumerate(inputs):
                    input_type = inp.get_attribute("type") or ""
                    placeholder = inp.get_attribute("placeholder") or ""
                    logger.info(f"  Input {i}: type={input_type}, placeholder={placeholder}")
                    if input_type in ("text", "email", "") or "email" in placeholder.lower():
                        inp.fill(email)
                        logger.info(f"  -> Filled input {i} with email")
                        break
            
            # Fill in password
            pwd_field = page.locator('input[placeholder="Password"]').first
            try:
                pwd_field.wait_for(state="visible", timeout=5000)
                pwd_field.fill(password)
                logger.info("Filled password field")
            except:
                # Fallback: try password type input
                logger.warning("Password placeholder failed, trying type=password...")
                pwd_fallback = page.locator('input[type="password"]').first
                pwd_fallback.fill(password)
                logger.info("Filled password via type=password")
            
            # Click sign in button
            logger.info("Clicking sign in...")
            try:
                sign_in_btn = page.locator('button:has-text("Sign in")').first
                sign_in_btn.wait_for(state="visible", timeout=5000)
                sign_in_btn.click()
                logger.info("Clicked Sign in button")
            except:
                # Fallback: try submit button
                logger.warning("Sign in text button failed, trying submit...")
                page.locator('button[type="submit"]').first.click()
            
            # Wait for redirect back to FPL
            logger.info("Waiting for login to complete...")
            page.wait_for_timeout(8000)
            
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
