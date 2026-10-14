/*
 * Copyright 2026 ObjectBox Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.services.semanticretrieval

import com.google.mediapipe.tasks.retrieval.components.VectorStore
import java.io.File

/**
 * [VectorStoreBenchmark] for [InMemoryVectorStore], an exact search: its recall must be 1.0, which
 * checks the benchmark itself. Runs with `./gradlew testDebugUnitTest -PvectorData=<dir>
 * --tests '*InMemoryVectorStoreBenchmark'`, where <dir> holds the data set's Android export.
 */
class InMemoryVectorStoreBenchmark : VectorStoreBenchmark() {
  override val dataDir: File? = System.getProperty("vectorData")?.let(::File)

  override fun createStore(name: String): VectorStore = InMemoryVectorStore()
}
