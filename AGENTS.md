# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Repository overview

This is ObjectBox's fork of Google AI Edge Gallery (`google-ai-edge/gallery`), an Android app that runs open LLMs on-device (LiteRT-LM, MediaPipe, ML Kit/AICore). The code is still upstream Google code under the `com.google.ai.edge.gallery` package.

## Purpose of this fork

**This fork exists to replace the app's SQLite storage with [ObjectBox](https://objectbox.io).** Most tasks in this repo serve that goal.

- **Main focus: vector search.** The app's semantic search currently stores embeddings in MediaPipe's `SqliteVectorStore` and searches them there. Replace that with ObjectBox and its HNSW vector search.
- **Likely next: all other storage.** We will probably replace all remaining SQLite storage as well. The details are still to be decided, so don't assume the scope beyond vector search is settled.

Where SQLite is used today (all of it through MediaPipe's `VectorStore` API):
- `services/semanticretrieval/SemanticRetrievalService.kt` creates the `SqliteVectorStore` and falls back to `InMemoryVectorStore`. The `VectorStore` interface (`customVectorStore` parameter) is the natural place to plug in ObjectBox. `SQLITE_RESERVED_METADATA_KEYS` is a workaround for a SQLite-specific bug.
- These callers each open their own `.db` file: `customtasks/smartalbum/` (`SemanticRetrievalServiceProvider`, `SmartAlbumIndexingWorker`, `SampleAlbumStorageManager`, which also deletes SQLite WAL/SHM files), `customtasks/videomomentfinder/VideoMomentFinderTask.kt`, and `services/semanticretrieval/GemmaEmbeddingModelStore.kt`.

The vector search baseline is pinned down by contract tests in `app/src/sharedTest/` (see Tests below). A replacement store must pass `VectorStoreContractTest`, and the service with it must pass `SemanticRetrievalServiceContractTest`. One SQLite quirk to know about: the raw `SqliteVectorStore.upsert` duplicates existing ids, and only `SemanticRetriever` hides this. A replacement store should replace existing ids itself.

Other app state is stored in Proto DataStore files, not SQLite (see Persistence below). Whether to move that to ObjectBox too is also still undecided.

(Paths are relative to `Android/src/app/src/main/java/com/google/ai/edge/gallery/`.)

Top-level layout:
- `Android/src/`: the Gradle project (single `:app` module). This is the only code that gets built.
- `model_allowlists/<version>.json`: model catalogs that the app downloads at runtime (see below). `model_allowlist.json` at the root is a legacy copy.
- `skills/`: example Agent Skills (`built-in/`, `featured/`), published to GitHub Pages by `.github/workflows/static.yml`. `skills/README.md` is the spec for the skill format (`SKILL.md` frontmatter, the `run_js` entry point `window.ai_edge_gallery_get_result`, `run_intent`, secrets).
- `mcp/README.md`: user docs for the MCP client integration.
- The vector test data (popular English Wikipedia pages embedded with EmbeddingGemma 2, reference queries with exact nearest neighbors) is built by a separate repo, `wikipedia-embedding-gemma-2` (`../wikipedia-embedding-gemma-2` locally). Its `data/export/android/` holds fvecs/ivecs files and a `manifest.json` for the on-device quality and performance tests.

## Build

Run all commands from `Android/src/`. CI (`.github/workflows/build_android.yaml`) uses JDK 21 and needs Android SDK platform `android-37.0`:

```bash
./gradlew assembleDebug      # debug APK
./gradlew assembleRelease    # what CI runs (signed with the debug key)
./gradlew installDebug       # install on a connected device
```

No lint or format task is configured. Existing code uses 2-space indentation in the ktfmt Google style, so keep new code in that style.

## Tests

Upstream ships no tests; all tests in this repo were added by the fork. There are three source sets:
- `app/src/test/`: JVM unit tests, no device needed. `unitTests.isReturnDefaultValues` is on, so framework calls like `android.util.Log` are no-ops instead of throwing.
- `app/src/androidTest/`: instrumentation tests. They need a running device or emulator, because MediaPipe's `SqliteVectorStore` and `SemanticRetriever` are native code built only for Android.
- `app/src/sharedTest/kotlin/`: compiled into both of the above. Contract tests and helpers live here (e.g. `FakeEmbedder` and `axisVector` in `TestVectors.kt`), so the same tests run against the JVM and the on-device implementations.

```bash
./gradlew testDebugUnitTest                                   # JVM tests
./gradlew testDebugUnitTest --tests '*SomeTest*'              # single JVM test class
./gradlew connectedDebugAndroidTest                           # instrumentation tests on a device/emulator
./gradlew connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.google.ai.edge.gallery.SomeTest  # single class
```

A headless emulator works for instrumentation tests: `emulator -avd <name> -no-window -no-audio -no-snapshot-save`, then `adb wait-for-device`. Use an x86_64 image with API 31 or newer.

To make Hugging Face model downloads work in a local build, replace the placeholders in `common/ProjectConfig.kt` (`clientId`, `redirectUri`) and `manifestPlaceholders["appAuthRedirectScheme"]` in `app/build.gradle.kts` (see `DEVELOPMENT.md`). The `google-services` plugin is declared with `apply false`, and the repo contains no `google-services.json`.

## Architecture

Paths below are relative to `Android/src/app/src/main/java/com/google/ai/edge/gallery/`.

**Tasks and models.** The home screen shows `Task`s (`data/Tasks.kt`), grouped into category tabs. Each task has a list of `Model`s (`data/Model.kt`). Every feature (LLM chat, Prompt Lab, Agent Chat, Mobile Actions, Tiny Garden, Smart Album, Video Moment Finder, Scrapbook…) is a `CustomTask` (`customtasks/common/CustomTask.kt`) that a Hilt module contributes with `@Provides @IntoSet`. The `*TaskModule.kt` files under `ui/` and `customtasks/` are these modules. `ModelManagerViewModel` receives the resulting `Set<CustomTask>`. To add a feature, follow `customtasks/examplecustomtask/`; its module is commented out on purpose.

**Model allowlist.** `ui/modelmanager/ModelManagerViewModel.kt` (the central, very large ViewModel for model state, downloads and init/cleanup) builds the model list as follows:
1. It first reads a local override at `/data/local/tmp/model_allowlist_test.json`. Push your own file there with `adb push` to test allowlist changes.
2. Otherwise it fetches `model_allowlists/<versionName with . replaced by _>.json` from the **upstream** `google-ai-edge/gallery` GitHub repo (`ALLOWLIST_BASE_URL`), not from this fork.
3. If the fetch fails, it falls back to the copy cached on disk.

When you bump `versionName` in `app/build.gradle.kts`, upstream needs a matching allowlist file. The allowlist entries map to models through `data/ModelAllowlist.kt`.

**Inference runtime.** `runtime/LlmModelHelper.kt` is the backend interface. `runtime/ModelHelperExt.kt` picks the implementation through `Model.runtimeHelper`: `AICoreModelHelper` for AICore models, `LlmChatModelHelper` (LiteRT-LM, in `ui/llmchat/`) for everything else. Tests can override the choice with `testingModelHelper`.

**Agent stack.** Agent Chat and multi-turn chat go through `agent/`:
- `AgentRuntimeExecutor` / `DefaultAgentRuntimeExecutor` run the agent and emit a stream of `AgentEvent`s.
- `agent/sessions/LlmSessionManager` handles session lifecycle, persists history and does context compaction (`SummarizationContextCompactor`).
- `tools/` contains the LLM tools (`run_js`, `run_intent`, `load_skill`, MCP tool calls), which go through `ToolDispatcher`.
- `skills/SkillManager` loads skills. `mcp/McpServersProvider` manages MCP servers.
- Native intents for `run_intent` are mapped in `intents/IntentHandler.kt`. `skills/README.md` gives the old `customtasks/agentchat/` path, which is out of date.
- Built-in skills ship in `app/src/main/assets/skills/`. This is a separate copy from the repo-root `skills/` directory.

**Persistence.** App state uses Proto DataStore. Schemas live in `app/src/main/proto/*.proto` (protobuf-lite, generated with the Gradle protobuf plugin), and the stores are provided in `di/AppModule.kt` (`settings.pb`, `user_data.pb`, `cutouts.pb`, …). Chat sessions are stored inside `UserData` (`data/ChatSessionRepository.kt`). Semantic search (`services/semanticretrieval/`) stores embeddings in MediaPipe's `SqliteVectorStore` and falls back to an in-memory store.

**DI and UI.** The app uses Hilt (`GalleryApplication`, `di/`) and Jetpack Compose with Navigation (`ui/navigation/GalleryNavGraph.kt`). Model downloads run as WorkManager jobs (`worker/DownloadWorker.kt`).
