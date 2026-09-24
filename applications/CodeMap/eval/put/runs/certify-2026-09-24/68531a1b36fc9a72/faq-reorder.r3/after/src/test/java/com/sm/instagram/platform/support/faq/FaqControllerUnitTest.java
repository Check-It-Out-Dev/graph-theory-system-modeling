package com.sm.instagram.platform.support.faq;

import com.sm.instagram.platform.support.faq.dtos.FaqReorderDtoIn;
import com.sm.instagram.platform.support.faq.services.FaqService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("FaqController Unit Tests")
class FaqControllerUnitTest {

    @Mock
    private FaqService faqService;

    @InjectMocks
    private FaqController controller;

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("reorderCategory delegates the category id and the ordered FAQ ids to the service and returns 204")
    void reorderCategoryDelegatesToServiceAndReturnsNoContent() {
        FaqReorderDtoIn dto = new FaqReorderDtoIn();
        dto.setFaqIds(List.of(30L, 10L, 20L));

        ResponseEntity<Void> response = controller.reorderCategory(7L, dto);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NO_CONTENT);
        assertThat(response.getBody()).isNull();

        ArgumentCaptor<Long> categoryIdCaptor = ArgumentCaptor.forClass(Long.class);
        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<Long>> faqIdsCaptor = ArgumentCaptor.forClass(List.class);
        verify(faqService).reorderCategory(categoryIdCaptor.capture(), faqIdsCaptor.capture());

        assertThat(categoryIdCaptor.getValue()).isEqualTo(7L);
        assertThat(faqIdsCaptor.getValue()).containsExactly(30L, 10L, 20L);
    }
}
