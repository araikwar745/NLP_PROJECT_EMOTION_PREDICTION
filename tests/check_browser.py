"""Run against a started app: python tests/check_browser.py [base URL]."""
import asyncio
import sys

from playwright.async_api import async_playwright, expect


async def check():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765")
        await expect(page.locator("#serverStatusText")).to_contain_text("ready", timeout=120000)
        text = page.locator("#textInput")
        button = page.locator("#analyzeBtn")
        result = page.locator("#resultSection")
        error = page.locator("#errorMsg")

        await text.fill("I love you so much")
        await button.click()
        await expect(result).to_be_visible()
        await expect(page.locator("#emotionWord")).to_have_text("Love")
        await expect(page.locator(".bar-row")).to_have_count(6)
        await page.wait_for_function("""() =>
            [...document.querySelectorAll('.bar-fill')]
                .some(bar => bar.getBoundingClientRect().width > 100)
        """)
        assert await page.locator(".bar-fill").first.evaluate(
            "bar => getComputedStyle(bar).display === 'block'"
        )
        print("PASS: Love prediction and six visible probability bars")

        for invalid in ("!!! ???", "asdkjh qwlekj zxcmnb", "मैं बहुत खुश हूँ"):
            await text.fill(invalid)
            await button.click()
            await expect(error).to_be_visible()
            await expect(result).to_be_hidden()
            await expect(button).to_be_enabled()
        await text.fill("   ")
        await expect(button).to_be_disabled()
        print("PASS: Invalid input clears the previous result; spaces cannot submit")

        # Hold one request so typing and Ctrl+Enter happen before its response.
        release = asyncio.Event()
        started = asyncio.Event()
        requests = []

        async def hold_prediction(route):
            requests.append(route.request.post_data)
            started.set()
            await release.wait()
            await route.continue_()

        await page.route("**/predict", hold_prediction)
        await text.fill("I love you so much")
        await button.click()
        await asyncio.wait_for(started.wait(), timeout=10)
        await expect(error).to_be_hidden()
        await text.fill("I feel so alone and hopeless today.")
        await text.press("Control+Enter")
        await expect(button).to_be_disabled()
        await expect(result).to_be_hidden()
        assert len(requests) == 1, "Typing/keyboard shortcut started an overlapping request"
        release.set()
        await expect(result).to_be_visible()
        await expect(page.locator("#echoedText")).to_contain_text("I love you so much")
        await page.unroute("**/predict", hold_prediction)
        await expect(button).to_be_enabled()
        await button.click()
        await expect(page.locator("#emotionWord")).to_have_text("Sadness")
        print("PASS: No overlapping requests; next analysis succeeds")

        await page.set_viewport_size({"width": 390, "height": 844})
        await expect(result).to_be_visible()
        assert await page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ), "Mobile layout overflows horizontally"
        assert not errors, errors
        print("PASS: Mobile layout and no JavaScript errors")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(check())
