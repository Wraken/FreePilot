# FreePilot

FreePilot is an attempt to create an homemade github copilot API.

## Installation

```bash
git clone https://github.com/Wraken/FreePilot.git
cd FreePilot
docker build -t freepilot .
docker run --gpus all -p 5000:5000 freepilot
```

## Usage
If the docker container is running you can test it with :
```bash
curl -s -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"n":1,"model":"fastertransformer","prompt":"def hello","max_tokens":50,"temperature":0.1,"stop":["\n\n"]}' http://127.0.0.1:5000/v1/engines/codegen/completions
```

To use this with github copilot vscode extension, you need to edit the `extension.js` file contained in :
- `C:\Users\%USERPROFILE%\.vscode\extensions\github.copilot-1.85.75\dist\extension.js` on windows
- `~/.vscode/extensions/github.copilot-1.85.75/dist/extension.js` on linux

Replace this :
- api.github.com/copilot_internal -> http://127.0.0.1:5000/copilot_internal
- copilot-proxy.githubusercontent.com -> http://127.0.0.1:5000

Don't forget to disable auto update of github copilot extension


Add this to vscode `settings.json`

```json
"github.copilot.advanced": {
        "model": "fastertransformer",
        "debug.overrideEngine": "codegen",
        "debug.testOverrideProxyUrl": "http://127.0.0.1:5000",
        "debug.overrideProxyUrl": "http://127.0.0.1:5000"
    },
```