package com.sm.instagram.platform.currency;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Entity
@NoArgsConstructor
@AllArgsConstructor
public class Currency {
    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "currency_generator")
    @SequenceGenerator(
        name = "currency_generator",
        sequenceName = "currency_seq",
        schema = "public",
        allocationSize = 50,
        initialValue = 1
    )
    private Long id;
    @Column(nullable = false)
    @NotBlank(message = "Name cannot be blank")
    @Size(max = 255, message = "Name cannot exceed 255 characters")
    private String name;

    @Column(nullable = false, unique = true)
    @NotBlank(message = "ISO code cannot be blank")
    @Size(max = 3, message = "ISO code cannot exceed 3 characters")
    private String isoCode;

    @Column(nullable = false)
    @NotBlank(message = "Sign cannot be blank")
    @Size(max = 3, message = "Sign cannot exceed 3 characters")
    private String sign;

    @Column(nullable = false)
    @Size(max = 3, message = "country code cannot exceed 3 characters")
    private String countryCode;
}

