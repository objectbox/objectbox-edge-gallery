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

import java.io.File
import org.junit.AfterClass

/** [SemanticRetrievalServiceContractTest] for the service using [ObjectBoxVectorStore] directly. */
class ObjectBoxSemanticRetrievalServiceJvmTest : SemanticRetrievalServiceContractTest() {
  override fun createService(databaseName: String, embedder: OnDeviceEmbedder) =
    SemanticRetrievalService(
      databaseName = databaseName,
      customEmbedder = embedder,
      customVectorStore = ObjectBoxVectorStore(File(directory, databaseName)),
    )

  companion object {
    private val directory = File(System.getProperty("java.io.tmpdir"), "objectbox-service-jvm-test")

    @JvmStatic
    @AfterClass
    fun deleteDirectory() {
      directory.deleteRecursively()
    }
  }
}
