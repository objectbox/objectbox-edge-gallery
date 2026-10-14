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
import java.io.File
import org.junit.AfterClass
import org.junit.runner.RunWith

/**
 * [SemanticRetrievalServiceContractTest] for the service with [ObjectBoxVectorStore] on a device,
 * where MediaPipe's [com.google.mediapipe.tasks.retrieval.semanticretriever.SemanticRetriever]
 * uses the store, as it would in the app.
 */
@RunWith(AndroidJUnit4::class)
class ObjectBoxSemanticRetrievalServiceDeviceTest : SemanticRetrievalServiceContractTest() {
  override fun createService(databaseName: String, embedder: OnDeviceEmbedder) =
    SemanticRetrievalService(
      databaseName = databaseName,
      context = InstrumentationRegistry.getInstrumentation().targetContext,
      customEmbedder = embedder,
      customVectorStore = ObjectBoxVectorStore(File(directory, databaseName)),
    )

  companion object {
    private val directory =
      File(
        InstrumentationRegistry.getInstrumentation().targetContext.filesDir,
        "objectbox-service-device-test",
      )

    @JvmStatic
    @AfterClass
    fun deleteDirectory() {
      directory.deleteRecursively()
    }
  }
}
