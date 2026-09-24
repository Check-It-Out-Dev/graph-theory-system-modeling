## Curation notes

Append-only. One line per partition decision taken on a product pull request (`/codemap …`), written
by the delta pipeline; the navigator reads them as the most recent word on where things live.

- 2026-09-16: (none yet — the first delta run writes the first line)
- 2026-09-16 backend@ff43730 (pack 1.0.1): ClearSecurityContextExtension.java → subsystem 205 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): ProbeListener.java → subsystem 205 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): AuthControllerTokenLoggingUnitTest.java → subsystem 10 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): LocalTotpCipherUnitTest.java → subsystem 6 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): InterruptsUnitTest.java → subsystem 0 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): LogSafeUnitTest.java → subsystem 0 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): junit-platform.properties → subsystem 205 — decided by RamzesX
- 2026-09-16 backend@ff43730 (pack 1.0.1): NEW subsystem [205] JUnit test-execution harness = ClearSecurityContextExtension.java, ProbeListener.java, junit-platform.properties — decided by RamzesX
- 2026-09-16 backend+frontend@ff43730 (pack 1.1.0): full reindex, 174 entities placed across 22 subsystems — decided by RamzesX
  - subsystem 0: OtpAuthUrlsUnitTest.java
  - subsystem 3: ActuatorExposureUnitTest.java, AuthFailureResponsesCustomizer.java, ExternalCredentialsAvailable.java, FreeFormMapSchemaCustomizer.java, GeoIpDatabaseHooks.java, GeoIpWorkDirectoryUnitTest.java, GoogleCloudClientLinkageUnitTest.java, GoogleCredentialsProviderFallbackUnitTest.java (+21 more)
  - subsystem 4: BasePatchProtectedFieldsUnitTest.java, PlatformService_LazyAssociation_IntegrationTest.java
  - subsystem 6: FirebaseEmulatorSeeder.java, ImprovedQRCodeServiceLoggingUnitTest.java, OAuthCallbackFailure.java, OtpAuthUrls.java, PublicProfileDiscriminatorUnitTest.java, TestAuthProvisionTotpUnitTest.java, TotpQRCodeStartupValidatorLoggingUnitTest.java, TwoFactorResponses.java
  - subsystem 7: AuthFailureResponsesCustomizerUnitTest.java, ErrorControllerNotPublishedUnitTest.java, ErrorEnvelopeResponsesCustomizer.java, ErrorEnvelopeResponsesCustomizerUnitTest.java, JwtAuthenticationFilterDevLitePublicUnitTest.java, LogForgeryUnitTest.java, MappingFailureStatusUnitTest.java, OneErrorShapeUnitTest.java (+4 more)
  - subsystem 8: SocialPostUrlValidatorUnitTest.java, ValidationPatternsUnitTest.java, VimeoUrlsValidatorUnitTest.java
  - subsystem 9: AuthControllerVerificationEmailUnitTest.java, SandboxActuatorSecurity.java, SandboxConfig.java, SandboxGuardFilter.java, SandboxPersonaPolicy.java, SandboxProperties.java
  - subsystem 10: FirebaseEmulator.java
  - subsystem 11: BoundaryRefusalUnitTest.java, CookieSameSitePolicyUnitTest.java, StripeWebhookControllerUnitTest.java
  - subsystem 15: TicketAccessTokenServiceUnitTest.java
  - subsystem 16: FreeFormMapSchemaCustomizerUnitTest.java, NullableFieldsAreDeclaredNullableUnitTest.java, OpenApiDateTimeFormatUnitTest.java, ResponseShapeMatchesReturnTypeUnitTest.java
  - subsystem 170: auth.contract.ts, sandbox-auth.service.spec.ts, sandbox-auth.service.ts, sandbox-persona-picker.component.html, sandbox-persona-picker.component.ts, sandbox-personas.ts, session-state.contract.ts, step-up.contract.ts (+1 more)
  - subsystem 171: applied-opportunities.contract.ts, avatar.component.ts, opportunities.contract.ts, opportunity-dictionaries.contract.ts, opportunity-dictionaries.service.spec.ts, reject-applicant-dialog.component.html, reject-applicant-dialog.component.ts
  - subsystem 172: cicd-runs-showcase.component.spec.ts, cicd-runs-showcase.component.ts, demo-export-zip.spec.ts, estate-map-showcase.component.spec.ts, estate-map-showcase.component.ts, graph-cost-data.ts, graph-topology-data.ts, graph-topology-showcase.component.spec.ts (+3 more)
  - subsystem 173: address.builder.ts, applied-opportunity.builder.ts, builders.spec.ts, codemap.fixture.ts, graph-topology.fixture.ts, index.ts, reject-applicant.fixture.ts, sandbox-host.component.spec.ts (+3 more)
  - subsystem 174: dialog-header.component.spec.ts, dialog-header.component.ts, legal.contract.ts, subscription.contract.ts
  - subsystem 175: support.contract.ts
  - subsystem 176: address.contract.ts, admin.contract.ts, cross-tab.ts, dictionary.contract.ts, preferences.contract.ts, registry.contract.ts, social-connections-settings.component.spec.ts, social-connections.contract.ts (+1 more)
  - subsystem 177: browser.ts, codemap-page.component.ts, dashboard-post-mock.component.ts, en.json, environment.sandbox.ts, handlers.ts, index-html.spec.ts, interactive-dashboard-preview.component.html (+18 more)
  - subsystem 178: checkout-sim.component.ts, demo-code.ts, demo-fixtures.account.spec.ts, demo.interceptor.spec.ts, glide.ts, guide-hint.spec.ts, guide-hint.ts, guide-runner.service.spec.ts (+5 more)
  - subsystem 205: application-e2e.properties, application-integration.properties, application-test-ratelimit.yml, application-test.properties, application.properties, logback-test.xml, testcontainers.properties
  - subsystem 206: codemap-page.component.html, codemap-page.component.spec.ts, codemap-recordings.ts, trajectory-player.component.html, trajectory-player.component.spec.ts, trajectory-player.component.ts
- 2026-09-16 backend+frontend@ff43730 (pack 1.1.0): NEW subsystem [206] codemap-trajectory-viewer = codemap-page.component.html, codemap-page.component.spec.ts, codemap-recordings.ts, trajectory-player.component.html, trajectory-player.component.spec.ts, trajectory-player.component.ts — decided by RamzesX
