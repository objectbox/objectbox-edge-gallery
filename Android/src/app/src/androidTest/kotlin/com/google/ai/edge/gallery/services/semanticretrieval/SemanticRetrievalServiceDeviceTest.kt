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
import org.junit.runner.RunWith

/**
 * Runs the service contract on a device, where it uses the same stack as the app: MediaPipe's
 * native SemanticRetriever on top of SqliteVectorStore.
 */
@RunWith(AndroidJUnit4::class)
class SemanticRetrievalServiceDeviceTest : SemanticRetrievalServiceContractTest() {
  override fun createService(databaseName: String, embedder: OnDeviceEmbedder) =
    SemanticRetrievalService(
      databaseName = databaseName,
      context = InstrumentationRegistry.getInstrumentation().targetContext,
      customEmbedder = embedder,
    )
}
