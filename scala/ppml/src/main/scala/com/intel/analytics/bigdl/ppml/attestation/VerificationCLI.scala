/*
 * Copyright 2016 The BigDL Authors.
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


package com.intel.analytics.bigdl.ppml.attestation

import org.apache.logging.log4j.LogManager
import scopt.OptionParser

import java.io.{BufferedOutputStream, BufferedInputStream};
import java.io.File;
import java.io.{FileInputStream, FileOutputStream};
import java.nio.file.{Paths, Path}
import java.util.Base64

import com.intel.analytics.bigdl.ppml.attestation.service._
import com.intel.analytics.bigdl.ppml.attestation.verifier._

/**
 * Simple Command Line tool to verify attestation service
 */
object VerificationCLI {
    def main(args: Array[String]): Unit = {

        val logger = LogManager.getLogger(getClass)
        case class CmdParams(appID: String = "test",
                             apiKey: String = "test",
                             attestationType: String = ATTESTATION_CONVENTION.MODE_EHSM_KMS,
                             attestationURL: String = "127.0.0.1:9000",
                             challenge: String = "test",
                             quotePath: String = "")

        val cmdParser: OptionParser[CmdParams] = new OptionParser[CmdParams](
          "PPML Quote Verification Cmd tool") {
            opt[String]('i', "appID")
              .text("app id for this app")
              .action((x, c) => c.copy(appID = x))
            opt[String]('k', "apiKey")
              .text("app key for this app")
              .action((x, c) => c.copy(apiKey = x))
            opt[String]('u', "attestationURL")
              .text("attestation service url, default is 127.0.0.1:9000")
              .action((x, c) => c.copy(attestationURL = x))
            opt[String]('t', "attestationType")
              .text("attestation service type, default is EHSMKeyManagementService")
              .action((x, c) => c.copy(attestationType = x))
            opt[String]('c', "challenge")
              .text("challenge to attestation service, default is '' which skip bi-attestation")
              .action((x, c) => c.copy(challenge = x))
            opt[String]('q', "quotePath")
              .text("quotePath, set to verify given quote instead of getting from service")
              .action((x, c) => c.copy(quotePath = x))
        }
        val params = cmdParser.parse(args, CmdParams()).get
        var quote = Array[Byte]()
        val quotePath = params.quotePath
        if (quotePath == "") {
          // Attestation Client
          val as = params.attestationType match {
              case ATTESTATION_CONVENTION.MODE_EHSM_KMS =>
                  new EHSMAttestationService(params.attestationURL.split(":")(0),
                      params.attestationURL.split(":")(1), params.appID, params.apiKey)
              case ATTESTATION_CONVENTION.MODE_DUMMY =>
                  new DummyAttestationService()
              case _ => throw new AttestationRuntimeException("Wrong Attestation service type")
          }

          val challengeString = params.challenge
          quote = params.attestationType match {
            case ATTESTATION_CONVENTION.MODE_EHSM_KMS =>
              Base64.getDecoder().decode(as.getQuoteFromServer(challengeString))
            case _ => throw new AttestationRuntimeException("Wrong Attestation service type")
          }

        } else {
          val tempQuotePath = Paths.get(quotePath).toAbsolutePath.normalize
          val quoteFile = tempQuotePath.toFile
          quoteFile.setExecutable(false)
          val in = new FileInputStream(quoteFile)
          val bufIn = new BufferedInputStream(in)
          quote = Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
          bufIn.close()
          in.close()
        }
        val quoteVerifier = new SGXDCAPQuoteVerifierImpl()
        quoteVerifier.verifyQuote(quote)
    }
}