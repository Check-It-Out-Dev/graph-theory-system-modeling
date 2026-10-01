package com.sm.instagram.platform.support.faq.services;

import com.sm.instagram.platform.common.exceptions.TranslatableException;
import com.sm.instagram.platform.support.faq.dtos.FaqReorderDtoIn;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.context.ApplicationContext;
import org.springframework.security.authentication.TestingAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowable;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyIterable;
import static org.mockito.Mockito.when;

/** Hidden acceptance test of the task faq-reorder (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class FaqReorderAcceptanceTest {

    private static final String KEY = "error.business.faq_reorder_mismatch";

    @Mock FaqRepository faqRepository;
    @Mock FaqCategoryRepository faqCategoryRepository;
    @Mock ApplicationContext applicationContext;
    @InjectMocks FaqService service;

    private final FaqCategory category = new FaqCategory();
    private final List<Faq> active = new ArrayList<>();

    @BeforeEach
    void setUp() {
        SecurityContextHolder.getContext().setAuthentication(new TestingAuthenticationToken("admin-uid", null));
        when(applicationContext.getBean(any(Class.class))).thenAnswer(inv -> service);
        category.setId(3L);
        active.add(faq(10L, 1));
        active.add(faq(11L, 2));
        active.add(faq(12L, 3));
        when(faqCategoryRepository.findById(3L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category)).thenReturn(active);
        when(faqRepository.findByCategoryAndActiveTrue(category)).thenReturn(active);
        when(faqRepository.findAllById(anyIterable())).thenReturn(active);
        when(faqRepository.findById(any())).thenAnswer(inv -> active.stream()
                .filter(f -> f.getId().equals(inv.getArgument(0))).findFirst());
        when(faqRepository.save(any(Faq.class))).thenAnswer(inv -> inv.getArgument(0));
        when(faqRepository.saveAll(anyIterable())).thenAnswer(inv -> inv.getArgument(0));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void theListedOrderBecomesTheDisplayOrder() {
        service.reorderCategory(3L, List.of(12L, 10L, 11L));
        assertThat(byId(12L).getDisplayOrder()).isEqualTo(1);
        assertThat(byId(10L).getDisplayOrder()).isEqualTo(2);
        assertThat(byId(11L).getDisplayOrder()).isEqualTo(3);
    }

    @Test
    void aMissingFaqIsRefusedWithTheKey() {
        assertMismatch(List.of(12L, 10L));
    }

    @Test
    void aForeignFaqIsRefusedWithTheKey() {
        assertMismatch(List.of(12L, 10L, 99L));
    }

    @Test
    void aRepeatedFaqIsRefusedWithTheKey() {
        assertMismatch(List.of(12L, 12L, 10L));
    }

    @Test
    void theRequestBodyCarriesTheIds() {
        FaqReorderDtoIn dto = new FaqReorderDtoIn();
        dto.setFaqIds(List.of(1L, 2L));
        assertThat(dto.getFaqIds()).containsExactly(1L, 2L);
    }

    @Test
    void theKeyIsTranslatedInBothLanguages() throws IOException {
        assertThat(bundle("messages_en.properties").getProperty(KEY)).isNotBlank();
        assertThat(bundle("messages_pl.properties").getProperty(KEY)).isNotBlank();
    }

    private void assertMismatch(List<Long> ids) {
        Throwable thrown = catchThrowable(() -> service.reorderCategory(3L, ids));
        assertThat(thrown).isInstanceOf(TranslatableException.class);
        assertThat(((TranslatableException) thrown).getMessageKey()).isEqualTo(KEY);
    }

    private Faq byId(Long id) {
        return active.stream().filter(f -> f.getId().equals(id)).findFirst().orElseThrow();
    }

    private Faq faq(Long id, int order) {
        Faq f = new Faq();
        f.setId(id);
        f.setCategory(category);
        f.setActive(true);
        f.setDisplayOrder(order);
        f.setQuestion("Q" + id);
        f.setAnswer("A" + id);
        return f;
    }

    private static Properties bundle(String name) throws IOException {
        Properties p = new Properties();
        try (InputStream in = FaqReorderAcceptanceTest.class.getClassLoader().getResourceAsStream(name)) {
            assertThat(in).as(name).isNotNull();
            p.load(new InputStreamReader(in, StandardCharsets.UTF_8));
        }
        return p;
    }
}
