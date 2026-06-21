# Visual Testing with Pytest and Screenshots

This project now includes a visual testing infrastructure that captures screenshots of the UI components during test execution. This allows for visual verification of the application state, similar to tools like Playwright, but adapted for the PySide6 desktop application.

## How it Works

1.  **Fixture**: A `take_screenshot` fixture is defined in `tests/conftest.py`.
2.  **Integration**: This fixture is injected into the tests in `tests/test_gui_flows.py`.
3.  **Capture**: At the end of each test (or critical steps), `take_screenshot(widget, "name")` is called.
4.  **Storage**: Screenshots are saved in the `tests/screenshots/` directory.

## Running Visual Tests

To run the tests and generate screenshots, simply execute the standard pytest command:

```bash
pytest tests/test_gui_flows.py
```

## Verifying Results

After running the tests, check the `tests/screenshots/` folder. You will find PNG images corresponding to each test case:

*   `tab1_data_management.png`
*   `tab2_strategy_config.png`
*   ...
*   `onboarding_wizard.png`

These images show the state of the widget at the time the screenshot was taken, allowing you to verify:
*   Layout correctness
*   Styling (Dark mode, colors)
*   Visibility of elements (e.g., Help text, Charts)

## Adding New Visual Tests

To add visual verification to a new test:

1.  Add `take_screenshot` to the test function arguments.
2.  Call `take_screenshot(your_widget, "unique_name")` at the point you want to capture.

```python
def test_my_new_feature(qapp, take_screenshot):
    widget = MyWidget()
    # ... perform actions ...
    take_screenshot(widget, "my_new_feature_state")
```
