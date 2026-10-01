package com.sm.instagram.platform.support.faq;

import com.sm.instagram.platform.support.faq.dtos.FaqReorderDtoIn;
import com.sm.instagram.platform.support.faq.services.FaqCategoryService;
import com.sm.instagram.platform.support.faq.services.FaqService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;

/**
 * Unit tests for {@link FaqCategoryController#reorderFaqs(Long, FaqReorderDtoIn)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("FaqCategoryController reorderFaqs")
class FaqCategoryControllerUnitTest {

    @Mock
    private FaqCategoryService faqCategoryService;

    @Mock
    private FaqService faqService;

    private FaqCategoryController controller;

    @BeforeEach
    void setUp() {
        controller = new FaqCategoryController(faqCategoryService, faqService);
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken("admin-42", null, List.of()));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("delegates the given category id and FAQ order to the service and returns 204")
    void delegatesToServiceAndReturnsNoContent() {
        FaqReorderDtoIn dto = new FaqReorderDtoIn();
        dto.setFaqIds(List.of(3L, 1L, 2L));

        ResponseEntity<Void> response = controller.reorderFaqs(7L, dto);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NO_CONTENT);
        ArgumentCaptor<List<Long>> captor = ArgumentCaptor.forClass(List.class);
        verify(faqService).reorderCategory(eq(7L), captor.capture());
        assertThat(captor.getValue()).containsExactly(3L, 1L, 2L);
    }
}
