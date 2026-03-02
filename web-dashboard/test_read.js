const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');

function readJSON(filename) {
    const filepath = path.join(DATA_DIR, filename);
    console.log(`Reading: ${filepath}`);
    try {
        if (!fs.existsSync(filepath)) {
            console.log("File does not exist");
            return null;
        }

        let content = '';
        let retries = 3;
        while (retries > 0) {
            try {
                content = fs.readFileSync(filepath, 'utf8');
                if (content && content.trim().length > 0) {
                    return JSON.parse(content);
                }
            } catch (err) {
                console.log(`Retry error: ${err.message}`);
            }
            retries--;
        }
        return null;
    } catch (e) {
        console.error(`Error: ${e.message}`);
        return null;
    }
}

const data = readJSON('multi_bot_state.json');
if (data) {
    console.log("Success!");
    console.log("Keys:", Object.keys(data));
    if (data.coin_states) {
        console.log("Coin count:", Object.keys(data.coin_states).length);
        if (data.coin_states['BTCUSDT']) {
             console.log("BTCUSDT found");
             console.log("OrderFlow:", JSON.stringify(data.coin_states['BTCUSDT'].orderflow_details, null, 2));
        } else {
             console.log("BTCUSDT MISSING in coin_states");
        }
    } else {
        console.log("coin_states MISSING");
    }
} else {
    console.log("Failed to read JSON");
}
