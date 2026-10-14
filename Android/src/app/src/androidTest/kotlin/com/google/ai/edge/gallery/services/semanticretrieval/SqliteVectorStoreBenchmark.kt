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

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.tasks.retrieval.components.SqliteVectorStore
import com.google.mediapipe.tasks.retrieval.components.VectorStore
import java.io.File
import org.junit.runner.RunWith

/**
 * [VectorStoreBenchmark] for MediaPipe's [SqliteVectorStore], the app's current vector store.
 *
 * Push the data set's Android export to the device and pass its location as a runner argument:
 * ```
 * adb push <export>/android/. /data/local/tmp/wikipedia-embedding-gemma-2/
 * ./gradlew connectedDebugAndroidTest \
 *   -Pandroid.testInstrumentationRunnerArguments.class=com.google.ai.edge.gallery.services.semanticretrieval.SqliteVectorStoreBenchmark \
 *   -Pandroid.testInstrumentationRunnerArguments.vectorData=/data/local/tmp/wikipedia-embedding-gemma-2
 * ```
 */
@RunWith(AndroidJUnit4::class)
class SqliteVectorStoreBenchmark : VectorStoreBenchmark() {
  override val dataDir: File? =
    InstrumentationRegistry.getArguments().getString("vectorData")?.let(::File)

  override val persistent = true

  private val directory =
    File(InstrumentationRegistry.getInstrumentation().targetContext.filesDir, "benchmark")

  // Like the app, pass an absolute path as the database name.
  override fun createStore(name: String): VectorStore =
    SqliteVectorStore(
      InstrumentationRegistry.getInstrumentation().targetContext,
      databaseFile(name).also { it.parentFile!!.mkdirs() }.absolutePath,
      TEST_DIMENSION,
    )

  override fun storageBytes(name: String): Long = sqliteFiles(name).sumOf { it.length() }

  override fun deleteStorage(name: String) {
    sqliteFiles(name).forEach { it.delete() }
  }

  private fun databaseFile(name: String) = File(directory, "$name.db")

  private fun sqliteFiles(name: String) =
    listOf("", "-wal", "-shm", "-journal").map { File(directory, "$name.db$it") }.filter { it.exists() }
}
