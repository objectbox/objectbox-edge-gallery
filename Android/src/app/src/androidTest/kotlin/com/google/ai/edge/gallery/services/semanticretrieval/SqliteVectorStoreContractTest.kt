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
import org.junit.runner.RunWith

/** Runs the [VectorStore] contract against MediaPipe's native SqliteVectorStore on a device. */
@RunWith(AndroidJUnit4::class)
class SqliteVectorStoreContractTest : VectorStoreContractTest() {
  override val persistsAcrossReopen = true

  // Upserting an existing id adds a second record with that id: get() and search() return both
  // and delete(ids) removes both. SemanticRetriever replaces the old record itself, so the app
  // never sees duplicates (covered by SemanticRetrievalServiceDeviceTest).
  override val upsertReplacesExistingId = false

  override fun createStore(name: String): VectorStore =
    SqliteVectorStore(
      InstrumentationRegistry.getInstrumentation().targetContext,
      "$name.db",
      TEST_DIMENSION,
    )
}
