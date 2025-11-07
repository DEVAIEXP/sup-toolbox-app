function isJsonString(str) {
    if (typeof str !== 'string' || str.trim() === '') {
        return false;
    }
    try {
        const result = JSON.parse(str);
        return typeof result === 'object' && result !== null;
    } catch (e) {
        return false;
    }
}

// Copied and adapted from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/javascript/extraNetworks.js#L582
function requestGet(url, data, handler, errorHandler) {
    if (typeof url !== 'string' || url.trim() === '') {
        console.error("requestGet: Invalid or empty URL provided.");
        if (typeof errorHandler === 'function') errorHandler(new Error("Invalid URL"));
        return;
    }

    const requestData = (data && typeof data === 'object') ? data : {};
    const successHandler = typeof handler === 'function' ? handler : () => { };
    const failureHandler = typeof errorHandler === 'function' ? errorHandler : (err) => { console.error("requestGet failed:", err); };

    let args = '';
    try {
        args = Object.keys(requestData).map(function (k) {
            const key = encodeURIComponent(k);
            const value = encodeURIComponent(requestData[k]);
            return `${key}=${value}`;
        }).join('&');
    } catch (e) {
        console.error("requestGet: Failed to serialize data object.", e);
        failureHandler(e);
        return;
    }

    const fullUrl = args ? `${url}?${args}` : url;

    const xhr = new XMLHttpRequest();
    xhr.open("GET", fullUrl, true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    const responseJson = isJsonString(xhr.responseText) ? JSON.parse(xhr.responseText) : {};
                    successHandler(responseJson);
                } catch (error) {
                    console.error("requestGet: Error parsing JSON response.", error);
                    failureHandler(error);
                }
            } else {
                failureHandler(new Error(`Request failed with status ${xhr.status}: ${xhr.statusText}`));
            }
        }
    };

    xhr.onerror = function () {
        failureHandler(new Error("Network error occurred."));
    };

    xhr.send();
}

// copied and adapted from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/javascript/ui.js#L348
function restart_ui() {
    const overlay = document.createElement('div');
    overlay.innerHTML = '<h1 style="font-family:monospace; margin-top:20%; color:lightgray; text-align:center;">Reloading...</h1>';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(20, 20, 20, 0.9)';
    overlay.style.zIndex = '10000';
    document.body.appendChild(overlay);

    const requestPing = function () {
        requestGet(
            window.location.href,
            {},
            function (data) {
                window.location.reload();
            },
            function (error) {
                console.log("Ping failed, retrying...");
                setTimeout(requestPing, 500);
            }
        );
    };

    setTimeout(requestPing, 2000);
}

/**
 * Moves all child elements from one or more source containers to their corresponding target containers.
 * By convention, the target ID for a source 'my-source-id' is expected to be 'my-source-id_target'.
 * After moving, the source containers are removed from the DOM by default.
 *
 * @param {string | string[]} source_ids - A single source element ID or an array of IDs.
 * @param {object} [options] - Optional settings.
 * @param {boolean} [options.remove_sources=true] - If true, removes the source elements after their content is moved.
 */
function reparent_flyout(source_ids, options = { remove_sources: true }) {
    // Ensure the input is always an array to simplify the logic.
    const sourceArray = Array.isArray(source_ids) ? source_ids : [source_ids];

    // Iterate over each provided source ID.
    sourceArray.forEach(source_id => {
        // Find the source element.
        const source = document.getElementById(source_id);
        
        // If the source element doesn't exist, we can't do anything for this ID.
        if (!source) {
            console.warn(`WARNING: Source element with ID '${source_id}' not found. Skipping.`);
            return; // 'return' inside a forEach acts like 'continue' in a for loop.
        }

        // Derive the target ID from the source ID
        const target_id = `${source_id}_target`;
        const target = document.getElementById(target_id);

        // If the corresponding target element doesn't exist, log an error.
        if (!target) {
            console.error(`ERROR: Reparenting for '#${source_id}' failed. Corresponding target '#${target_id}' not found.`);
            return;
        }

        // Move each child node from the source to the target.
        // The `while (source.firstChild)` loop is efficient because `appendChild`
        // automatically removes the child from its original parent.
        while (source.firstChild) {
            target.appendChild(source.firstChild);
        }

        // Remove the now-empty source container if the option is enabled.
        if (options.remove_sources) {
            source.remove();
        }
        
        console.log(`SUCCESS: Content from '#${source_id}' has been reparented to '#${target_id}'.`);
    });
}

function position_flyout(anchorId, target_id) {
    if (!anchorId) { return; }
    
    const anchorElem = document.getElementById(anchorId);
    const flyoutElem = document.getElementById(target_id);
    
    if (anchorElem && flyoutElem) {
        //console.log("JS: Positioning flyout relative to:", anchorId);
        const anchorRect = anchorElem.getBoundingClientRect();
        const flyoutWidth = flyoutElem.offsetWidth;
        const flyoutHeight = flyoutElem.offsetHeight;

        let topPosition = anchorRect.top + (anchorRect.height / 2) - (flyoutHeight / 2);
        let leftPosition = anchorRect.left + (anchorRect.width / 2) - (flyoutWidth / 2);
        
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        if (leftPosition < 8) leftPosition = 8;
        if (topPosition < 8) topPosition = 8;
        if (leftPosition + flyoutWidth > windowWidth) leftPosition = windowWidth - flyoutWidth - 8;
        if (topPosition + flyoutHeight > windowHeight) topPosition = windowHeight - flyoutHeight - 8;

        flyoutElem.style.top = `${topPosition}px`;
        flyoutElem.style.left = `${leftPosition}px`;
    }
}

// This is the new main function called by Gradio's .then() event
function update_flyout_from_state(jsonData) {
    //console.log("JS: update_flyout_from_state() called with data:", jsonData);
    
    if (!jsonData) return;
 
    const state = JSON.parse(jsonData);
    const { isVisible, anchorId, targetId } = state;
    const flyout = document.getElementById(targetId);

    if (!flyout) {
        console.error("ERROR: Cannot update UI. Flyout container not found.");
        return;
    }
    //console.log("JS: Parsed state:", { isVisible, anchorId });
    if (isVisible) {
        flyout.style.display = 'flex';
        position_flyout(anchorId, targetId);
    } else {
        flyout.style.display = 'none';
    }
}