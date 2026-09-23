package com.sm.instagram.platform.support.ticket.models;

import jakarta.persistence.Version;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.assertj.core.api.Assertions.assertThat;

/** Hidden acceptance test of the task support-ticket-version (prompt under test, instance backend-conventions). */
class SupportTicketVersionAcceptanceTest {

    @Test
    void theEntityCarriesAnOptimisticLockVersion() throws Exception {
        Field version = SupportTicket.class.getDeclaredField("version");
        assertThat(version.getType()).isIn(Long.class, long.class);
        assertThat(version.isAnnotationPresent(Version.class)).isTrue();
    }

    @Test
    void aChangesetInTheMasterChangelogAddsTheVersionColumn() throws IOException {
        Pattern sql = Pattern.compile("(?is)alter\\s+table\\s+(public\\.)?\"?support_ticket\"?\\s+add\\s+(column\\s+)?(if\\s+not\\s+exists\\s+)?\"?version\"?\\b");
        Pattern xml = Pattern.compile("(?is)<addColumn[^>]*tableName=\"support_ticket\"[^>]*>.*?<column[^>]*name=\"version\"");
        boolean found = false;
        for (String file : includedChangesets()) {
            String text = resource("db/changelog/" + file);
            if (text != null && (sql.matcher(text).find() || xml.matcher(text).find())) {
                found = true;
            }
        }
        assertThat(found).as("an included changeset adds support_ticket.version").isTrue();
    }

    private static List<String> includedChangesets() throws IOException {
        String master = resource("db/changelog/changelog.xml");
        assertThat(master).isNotNull();
        List<String> files = new ArrayList<>();
        Matcher m = Pattern.compile("<include\\s+file=\"([^\"]+)\"").matcher(master.replaceAll("(?s)<!--.*?-->", ""));
        while (m.find()) {
            files.add(m.group(1));
        }
        return files;
    }

    private static String resource(String path) throws IOException {
        try (InputStream in = SupportTicketVersionAcceptanceTest.class.getClassLoader().getResourceAsStream(path)) {
            return in == null ? null : new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
