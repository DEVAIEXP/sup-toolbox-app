# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-10

This marks the first major stable release after a significant architectural refactor, focusing on robustness, maintainability, and developer experience.

### Added

-   **Modular Architecture:**
    -   Introduced a clean, multi-file architecture separating responsibilities:
        -   `app.py`: The main application orchestrator.
        -   `ui_layout.py`: Defines the complete UI structure and components.
        -   `ui_events.py`: Contains all event handling logic and business rules.
        -   `ui_state.py`: Provides a centralized `AppState` dataclass for managing shared application state.
-   **State Management:**
    -   Implemented a shared `AppState` container to manage application state (e.g., `uidata`, `cancel_event`, pipeline instances) cleanly, removing all global variables.
    -   Event logic is now encapsulated within an `EventHandlers` class, which receives the shared state and UI components via dependency injection.
-   **Developer Experience:**
    -   Created a `UIComponents` dataclass to hold all Gradio component instances, providing full static type checking and editor Intellisense (auto-completion) for component access.
    -   Refactored event bindings in `app.py` to use direct attribute access (e.g., `c.run_btn.click`) instead of error-prone string-based dictionary keys.

### Changed

-   **Event Handling Logic:**
    -   Reverted the primary image generation event from using the `@livelog` decorator back to the original, more robust `threading` and `queue` pattern. This ensures the progress bar and logs update correctly even when the browser tab is not in focus.
    -   The `generate` method now acts as a generator that spawns a `process_image` worker thread, yielding updates from a queue to the UI.
-   **Event Binding:**
    -   The main application class (`GradioApp`) is now solely responsible for orchestrating the app's lifecycle: initializing state, building the UI, and binding events. All implementation logic has been moved to `EventHandlers`.
    -   Simplified event binding calls by removing unnecessary `lambda` and `partial` wrappers where methods could be called directly. `partial` is now only used for its intended purpose of pre-filling static arguments.
-   **Code Quality & Style:**
    -   Integrated `ruff` as the primary tool for linting and formatting, replacing `isort` and `black`.
    -   Refactored code to resolve all major `ruff` warnings, including:
        -   Replacing unnecessary list comprehensions with more efficient set comprehensions (e.g., `set([...])` -> `{...}`).
        -   Replacing list comprehensions inside `any()` with memory-efficient generator expressions.
        -   Eliminating all "bare `except`" blocks by specifying `except Exception` to prevent hiding critical system errors.

### Fixed

-   **Component State Synchronization:**
    -   Fixed a critical bug where event handlers were reading stale state from component instances instead of receiving the latest values from the UI. All event handler methods now correctly receive up-to-date component values as arguments from the `inputs` list of the event trigger.
-   **Gradio Context Errors:**
    -   Resolved `AttributeError: Cannot call .click outside of a gradio.Blocks context` by centralizing the `gr.Blocks` context management within the `GradioApp` constructor, ensuring both UI creation and event binding occur within the same active context.
-   **`gr.EventData` Injection:**
    -   Fixed a bug where `gr.EventData` was being passed as `None` to event handlers. Implemented a robust wrapper pattern for event callbacks that require `EventData`, ensuring Gradio's event data injection mechanism works correctly with the class-based architecture.
-   **Preset Loading Logic:**
    -   Corrected a bug in the `load_preset` function where it was mutating the global `RESTORER_CONFIG_MAPPING` dictionary with component instances instead of classes, which caused a `TypeError` on subsequent UI interactions. The function now correctly updates the application state without modifying the original class mapping.